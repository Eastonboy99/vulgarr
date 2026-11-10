"""Remote NudeNet GPU worker for vulgarr (DockerStacks).

Listens for POST /detect with a multipart "frame" JPEG, runs NudeNet on the
GPU this container was assigned (NVIDIA_VISIBLE_DEVICES), and returns the raw
detection list in the shape vulgarr's patched classifier expects:

    {"detections": [{"class": "...", "score": 0.99, "box": [...]}, ...]}

The model (~70MB) is downloaded to /models on first use and cached there.
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from nudenet import NudeDetector
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="vulgarr-remote-classifier")

# One detector per process; a process is pinned to a single GPU via
# NVIDIA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES in compose.
_cache: dict[str, NudeDetector] = {}


def detector() -> NudeDetector:
    det = _cache.get("det")
    if det is None:
        det = NudeDetector()
        _cache["det"] = det
    return det


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/detect")
async def detect(frame: UploadFile = File(...)) -> dict:
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
