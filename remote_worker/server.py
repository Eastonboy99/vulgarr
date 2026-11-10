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

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import json
import re
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
        detections = await run_in_threadpool(lambda: detector().detect(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"detections": detections or []}


class EncodeParams(BaseModel):
    source_codec: str
    crf: int = Field(ge=0, le=51)
    preset: str
    # The host supplies its canonical build_blur_filter output. This avoids
    # a second subtly divergent implementation of enable-window batching.
    filter_complex: str = Field(max_length=100000)


_encode_slot = asyncio.Semaphore(1)


def _encode_segment(source: Path, output: Path, params: EncodeParams):
    # The worker is private-network only, but accepting arbitrary ffmpeg
    # graphs would still allow movie=/etc/... or network-source filters. Only
    # accept the host's boxblur grammar, never general ffmpeg expressions.
    label = r'(?:0:v|vb(?:_stage\d+)?)'
    number = r'\d+\.\d{3}'
    term = rf'between\(t,{number},{number}\)'
    stage = (rf'\[{label}\]boxblur=luma_radius=\d+:luma_power=\d+:'
             rf"chroma_radius=\d+:chroma_power=\d+:enable='{term}(?:\+{term})*'\[{label}\]")
    if not re.fullmatch(rf'(?:{stage})(?:;{stage})*|\[0:v\]null\[vb\]', params.filter_complex):
        raise ValueError('Unsupported blur graph')
    encoder = {'hevc': 'hevc_nvenc', 'h265': 'hevc_nvenc', 'h264': 'h264_nvenc'}.get(params.source_codec)
    if encoder is None:
        raise ValueError('Source codec has no compatible NVENC mapping')
    # NVENC presets and quality controls are not libx26x's. Use constant
    # quality VBR, not an invalid -crf or a CPU preset like veryfast.
    preset = {'ultrafast': 'p1', 'superfast': 'p2', 'veryfast': 'p3',
              'faster': 'p4', 'fast': 'p4', 'medium': 'p5',
              'slow': 'p6', 'slower': 'p7', 'veryslow': 'p7'}.get(params.preset, 'p5')
    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(source),
           '-filter_complex', params.filter_complex, '-map', '[vb]',
           '-c:v', encoder, '-preset', preset, '-rc', 'vbr', '-cq', str(params.crf),
           '-b:v', '0', '-an', '-sn', '-f', 'matroska', str(output)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    if result.returncode or not output.exists() or not output.stat().st_size:
        raise RuntimeError('NVENC encode failed')


@app.post('/encode')
async def encode(segment: UploadFile = File(...), params: str = Form(...)):
    """Private-network segment encoder; CPU-only workers return 503.

    The host retries its original codec-matched software path on *any* failure.
    A successful encode is not a splice-compatibility guarantee: the host must
    validate HEVC parameter state and retain final concat decode verification.
    """
    try:
        options = EncodeParams.model_validate(json.loads(params))
    except Exception:
        raise HTTPException(422, 'Invalid encode parameters')
    async with _encode_slot:
        tmp = tempfile.TemporaryDirectory()
        directory = Path(tmp.name)
        try:
            source = directory / 'segment.mkv'
            with source.open('wb') as handle:
                total = 0
                while chunk := await segment.read(1024 * 1024):
                    total += len(chunk)
                    if total > 4 * 1024**3:
                        raise ValueError('Segment exceeds 4 GiB')
                    await run_in_threadpool(handle.write, chunk)
            output = directory / 'encoded.mkv'
            started = time.monotonic()
            _stats['in_flight'] += 1
            try:
                await run_in_threadpool(_encode_segment, source, output, options)
                _stats['requests_served'] += 1
                _stats['total_latency_seconds'] += time.monotonic() - started
            finally:
                _stats['in_flight'] -= 1
            return FileResponse(output, media_type='video/x-matroska',
                                background=BackgroundTask(tmp.cleanup))
        except Exception:
            tmp.cleanup()
            raise HTTPException(503, 'Remote segment encode unavailable')


class ScanRequest(BaseModel):
    source_url: str
    start: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(gt=0, le=300, allow_inf_nan=False)
    interval: float = Field(gt=0, allow_inf_nan=False)


# Two bounded requests permit decode N+1 to overlap ONNX inference N. Only
# source decoding is serialized, not the whole extraction+classification job.
_scan_slot = asyncio.Semaphore(2)
_decode_lock = threading.Lock()
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
        with _decode_lock:
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
