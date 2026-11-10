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

## Chunk extraction on the worker

Set these in the **vulgarr** container (both compose examples forward them):

```dotenv
SPF_CLASSIFIER_URLS=http://gpu-host:7861,http://gpu-host:7862
SPF_REMOTE_EXTRACT=1
SPF_INTERNAL_BASE_URL=http://vulgarr-host:8011
SPF_INTERNAL_TOKEN=replace-with-a-long-random-secret
```

Set this on the **GPU box**, before starting its compose stack:

```dotenv
SPF_SOURCE_BASE_URL=http://vulgarr-host:8011
```

The base URL must be reachable from the worker container and refer to vulgarr,
not the GPU box. The worker restricts source reads to this exact scheme/host/port
and `/internal/source`. Do not put a trailing path prefix in either base URL.
The token is kept only on vulgarr; it signs short-lived URLs scoped to one exact
file. GET/HEAD and single byte ranges (including suffix ranges) are supported,
so ffmpeg can seek without uploading a whole movie. A URL is itself a bearer
credential: use TLS or a trusted, firewalled network, and avoid logging its query
string in proxies/access logs. Worker endpoints are private-network APIs, not
Internet-facing services. Never publish worker ports without network controls.

`POST /scan` reads at most 300 seconds per request, attempts CUDA/NVDEC decode,
then retries decoding on CPU if CUDA fails. Incomplete/malformed score responses
are rejected as a whole before any of that chunk enters the checkpoint. Failure
retries local extraction plus remote batch inference, then local CPU inference.
A missing transport setting leaves the original local-extraction path enabled.

Local extraction is double-buffered: chunk N+1 decodes while N is classified,
with at most two temporary chunk directories alive. Remote scans similarly
prefetch the next bounded request while the previous scores are checkpointed.
Each worker permits two chunk requests but serializes source decoding: chunk
N+1 can extract while N runs inference, without two concurrent source decodes.
Use one process/container per GPU. Every successful frame is appended separately
to `scores.jsonl`; resume still trusts only its contiguous prefix, not its maximum
timestamp. Interrupted or malformed chunks never advance that prefix.

Configured remote-extraction scans skip the whole-job heavy pipeline gate. A
local extraction fallback reacquires it for that decode, preserving audio/blur
exclusion. This removes local decode CPU contention, **not** the underlying NFS
traffic: the HTTP source still reads the media library on the vulgarr host.

## Batch inference and observability

- `POST /detect-batch`: multipart `frames` (1–16 JPEGs), returns
  `{"detections": [[...], [...]]}` in input order. Vulgarr sends batches of 12;
  failed/malformed batches retry the existing single-frame `/detect` route.
- `GET /health`: liveness plus process-local statistics. Model loading is lazy;
  before first inference, the device is reported as `not loaded`.
- `GET /stats`: device/providers, successful requests and frames served,
  mean successful-request latency in milliseconds, and current in-flight count.
  Counts reset on worker restart; failures do not count as successful requests.
- The queue page polls worker health/statistics about every five seconds. Scan
  messages report phase, classified frames/second and in-flight client requests.
  `remote` counts successful remote frames; `fallback` counts frames requiring
  degraded extraction or local inference (a frame can contribute to both if
  extraction fell back but remote inference succeeded).

Fresh databases default to 16 in-flight classification requests when
`SPF_CLASSIFIER_URLS` is set, otherwise 8. Saved settings are never overwritten:
existing installations should set **Frame classify concurrency** to 16 in
Settings if desired. Batch concurrency bounds requests, not individual frames.

## Optional NVENC segment encoding

```dotenv
# On vulgarr; a worker base URL, without /encode:
SPF_REMOTE_ENCODE_URL=http://gpu-host:7861
```

`POST /encode` accepts multipart `segment` (a video-only MKV) and a JSON `params`
form field (`source_codec`, `crf`, `preset`, `filter_complex`). The host sends the
exact `build_blur_filter` graph, including its enable-window batching; the worker
accepts only that restricted boxblur/null grammar, not arbitrary ffmpeg filters.
Only H.264/HEVC map to NVENC; other source codecs keep their existing local
encoder. CPU presets map to NVENC p1–p7; CRF maps to constant-quality VBR (`cq`),
which is not numerically identical to libx26x CRF. Segment uploads are capped at
4 GiB. The image installs ffmpeg and requests NVIDIA `video` driver capabilities
for NVDEC/NVENC; check the deployed image with `ffmpeg -encoders` and a real
encode because the available GPU/driver may still lack a requested capability.
CPU-only workers return 503 for encoding; classification continues to work.

**Codec identity alone does not establish splice safety.** Vulgarr rejects
remote output when codec/profile/pixel format/resolution/color/frame-rate
metadata differ, when uploaded segment timestamp/duration drift is detected,
or when traced HEVC SPS/PPS derived-state fields differ or cannot be probed.
NVENC cannot reproduce every x265 knob, so HEVC offload may legitimately fall
back on an otherwise healthy worker. A returned MKV must decode cleanly before
it becomes a completed segment. The existing concatenation, whole-output decode
verification, matched-x265 CPU encoder and whole-file fallback remain intact.
Any transport/encode/validation failure retries the original local encoder.

### Deployment validation (required before enabling production encode offload)

The unit suite uses mocked inference/ffmpeg and does **not** establish real NVENC
splice compatibility. On the actual GPU and representative H.264/HEVC media:

1. Apply one reviewed blur with remote encode enabled; confirm the queue reports
   `NVENC` rather than CPU fallback and keep the original source untouched.
2. Decode the entire resulting Vulgarr Edit with
   `ffmpeg -v error -xerror -i "Title - Vulgarr Edit.mkv" -f null -`.
3. Inspect both splice boundaries and the precise blur on/off windows visually,
   including audio sync. Repeat for 8/10-bit HEVC, sparse GOPs and HDR sources.
4. Shut the worker down mid-scan and mid-encode; confirm local fallback, then
   restart vulgarr mid-scan to verify checkpoint continuation.
5. If any mismatch/corruption appears, unset `SPF_REMOTE_ENCODE_URL`; do not
   relax the compatibility checks to make a particular GPU pass.
