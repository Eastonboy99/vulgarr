"""Remote NudeNet GPU worker for vulgarr (DockerStacks).

Listens for POST /detect with a multipart "frame" JPEG, runs NudeNet on the
GPU this container was assigned (NVIDIA_VISIBLE_DEVICES), and returns the raw
detection list in the shape vulgarr's patched classifier expects:

    {"detections": [{"class": "...", "score": 0.99, "box": [...]}, ...]}

The model (~70MB) is downloaded to /models on first use and cached there.
"""

import tempfile
import asyncio
import os
import math
import subprocess
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from nudenet import NudeDetector
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="vulgarr-remote-classifier")

# One detector per process; a process is pinned to a single GPU via
# NVIDIA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES in compose.
_cache: dict[str, NudeDetector] = {}
_load_lock = threading.Lock()
_stats = {"requests_served": 0, "frames_served": 0, "total_latency_seconds": 0.0, "in_flight": 0}


def detector() -> NudeDetector:
    det = _cache.get("det")
    if det is None:
        with _load_lock:
            det = _cache.get("det")
            if det is None:
                det = NudeDetector()
                _cache["det"] = det
    return det


@app.get("/health")
async def health() -> dict:
    return {"ok": True, **await stats()}


@app.post("/detect")
async def detect(frame: UploadFile = File(...)) -> dict:
    started = time.monotonic()
    _stats['in_flight'] += 1
    try:
        result = await _detect_upload(frame)
        _stats['requests_served'] += 1
        _stats['frames_served'] += 1
        _stats['total_latency_seconds'] += time.monotonic() - started
        return result
    finally:
        _stats['in_flight'] -= 1


async def _detect_upload(frame: UploadFile) -> dict:
    data = await frame.read()
    if not data:
        return {"detections": []}
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        tmp_path = str(Path(tmp.name))
    try:
        # Inference is synchronous and can take tens of ms -- run it in the
        # threadpool so concurrent requests don't serialize on (and time out
        # behind) the event loop. ONNX Runtime sessions are thread-safe for
        # concurrent .run() calls.
        detections = await run_in_threadpool(detector().detect, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"detections": detections or []}


class ScanRequest(BaseModel):
    source_url: str
    start: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(gt=0, le=300, allow_inf_nan=False)
    interval: float = Field(gt=0, allow_inf_nan=False)


_scan_slot = asyncio.Semaphore(1)
_EXPLICIT = {'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
             'MALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
             'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED'}


def _scan_chunk(params: ScanRequest) -> list[dict]:
    """Decode once per bounded chunk; retry CUDA decode on CPU from scratch.

    Do not return a partial extraction after hardware decode failed: it could
    look like a successful empty tail and poison the caller's checkpoint.
    HTTP source capabilities are restricted to a configured vulgarr origin,
    preventing this endpoint from becoming an arbitrary URL fetch proxy.
    """
    trusted = urlsplit(os.getenv('SPF_SOURCE_BASE_URL', ''))
    supplied = urlsplit(params.source_url)
    if (trusted.scheme not in {'http', 'https'} or supplied.scheme != trusted.scheme
            or supplied.netloc != trusted.netloc or supplied.path != '/internal/source'):
        raise ValueError('Source must use SPF_SOURCE_BASE_URL origin and /internal/source')
    if math.ceil(params.duration / params.interval) > 10000:
        raise ValueError('Too many frames in chunk')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for hardware in (True, False):
            for frame in directory.glob('*.jpg'):
                frame.unlink()
            cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y']
            if hardware:
                cmd += ['-hwaccel', 'cuda']
            cmd += ['-ss', str(params.start), '-rw_timeout', '30000000',
                    '-i', params.source_url, '-t', str(params.duration),
                    '-vf', f'fps=1/{params.interval},format=yuvj420p',
                    '-q:v', '4', str(directory / 'frame_%08d.jpg')]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if result.returncode == 0:
                break
        else:
            raise RuntimeError('Both CUDA and CPU extraction failed')
        frames = sorted(directory.glob('*.jpg'))
        # Mirror fps sampling: one score per nominal sample. Missing frames
        # trigger a client retry, never a silently successful short checkpoint.
        expected = max(1, round(params.duration / params.interval))
        if len(frames) != expected:
            raise RuntimeError('Incomplete decoded chunk')
        det = detector()
        return [{'t': params.start + i * params.interval,
                 'c': max((d['score'] for d in det.detect(str(frame))
                           if d.get('class') in _EXPLICIT), default=0.0)}
                for i, frame in enumerate(frames)]


@app.post('/scan')
async def scan(params: ScanRequest):
    async with _scan_slot:
        started = time.monotonic()
        _stats['in_flight'] += 1
        try:
            scores = await run_in_threadpool(_scan_chunk, params)
            _stats['requests_served'] += 1
            _stats['frames_served'] += len(scores)
            _stats['total_latency_seconds'] += time.monotonic() - started
            return {'scores': scores}
        except Exception:
            # Do not echo signed source URLs (including in subprocess errors).
            raise HTTPException(503, 'Remote extraction failed')
        finally:
            _stats['in_flight'] -= 1


@app.get('/stats')
async def stats() -> dict:
    """Process-local counters; querying health never loads the model."""
    det = _cache.get('det')
    session = getattr(det, 'onnx_session', None)
    providers = session.get_providers() if session is not None else []
    count = _stats['requests_served']
    return {
        **_stats,
        'device': ('GPU' if 'CUDAExecutionProvider' in providers else 'CPU') if det else 'not loaded',
        'providers': providers,
        'avg_latency_ms': 1000 * _stats['total_latency_seconds'] / count if count else 0,
    }


@app.post('/detect-batch')
async def detect_batch(frames: list[UploadFile] = File(...)) -> dict:
    """Keep input order, and reject oversize batches rather than partial success.

    Inference remains off the event loop. Sequential frames within each request
    avoid oversubscribing an ONNX session; clients overlap bounded requests.
    A failed request can safely be retried frame-by-frame by older clients.
    """
    if not 1 <= len(frames) <= 16:
        raise HTTPException(422, 'Expected between 1 and 16 frames')
    started = time.monotonic()
    _stats['in_flight'] += 1
    try:
        results = [(await _detect_upload(frame))['detections'] for frame in frames]
        _stats['requests_served'] += 1
        _stats['frames_served'] += len(frames)
        _stats['total_latency_seconds'] += time.monotonic() - started
        return {'detections': results}
    finally:
        _stats['in_flight'] -= 1
