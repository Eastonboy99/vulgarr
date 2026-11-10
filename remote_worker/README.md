# Remote GPU classifier worker

Offloads vulgarr's per-frame NudeNet scene-scan classification to a machine
with a GPU (or several). Run this compose file on the GPU box:

    cd remote_worker && docker compose up -d --build

Workers listen on :7861 (GPU 0) and :7862 (GPU 1). Then point vulgarr at them:

    SPF_CLASSIFIER_URLS=http://<gpu-host>:7861,http://<gpu-host>:7862

Frames are distributed round-robin across all listed workers. Any worker
failure (down, timeout, bad response) transparently falls back to local CPU
classification, so the GPU box being powered off never breaks a scan -- it
just runs slow again.

Notes:
- The base image carries CUDA 12.4 + cuDNN runtime; requires the NVIDIA
  container toolkit. Without a GPU, onnxruntime falls back to CPU (still
  works as a remote CPU offloader).
- The `/detect` endpoint runs inference in a threadpool -- ONNX sessions are
  safe for concurrent `.run()` calls, and this keeps concurrent requests from
  serializing on the event loop (which showed up as dropped connections when
  vulgarr's classify concurrency was 8+).
