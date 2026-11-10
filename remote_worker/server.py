"""Remote NudeNet GPU worker for vulgarr (DockerStacks).

Listens for POST /detect with a multipart "frame" JPEG, runs NudeNet on the
GPU this container was assigned (NVIDIA_VISIBLE_DEVICES), and returns the raw
detection list in the shape vulgarr's patched classifier expects:

    {"detections": [{"class": "...", "score": 0.99, "box": [...]}, ...]}

The model (~70MB) is downloaded to /models on first use and cached there.
"""

import tempfile
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
