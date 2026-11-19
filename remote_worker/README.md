# Remote GPU classifier worker

Offloads vulgarr's per-frame NudeNet scene-scan classification to a machine
with a GPU (or several). Run this compose file on the GPU box:

    cd remote_worker && docker compose up -d --build

Workers listen on :7861 (GPU 0) and :7862 (GPU 1). Then point vulgarr at them:

    SPF_CLASSIFIER_URLS=http://<gpu-host>:7861,http://<gpu-host>:7862

Frames are distributed across healthy workers. Failed workers are quarantined
for five minutes, with one client request admitted per worker URL. Configured
remote outages permit only bounded CPU fallback (64 frames / 30 cumulative
seconds); exhausting that budget withholds the new scene generation and retains
the prior successful edit, rather than silently spending hours on CPU. A title
also has a twelve-failed-attempt remote budget. Intentional CPU-only mode, without
classifier URLs, retains the normal local path. See the
[throughput controls](../docs/SCENE_THROUGHPUT_IMPLEMENTATION.md).

Notes:
- The base image carries CUDA 12.4 + cuDNN runtime; requires the NVIDIA
  container toolkit. Without a GPU, onnxruntime falls back to CPU (still
  works as a remote CPU offloader).
- Inference stays off the event loop, but a bounded process-wide guard serializes
  the shared detector across every inference route. Provider/CUDA failures
  quarantine inference until operator recovery; no automatic restart is done.
  This is defensive admission, not proof of the original CUDA fault's cause.

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
The server still permits two chunk requests and serializes source decoding;
the current client conservatively admits one request per worker URL. Prefetch
can overlap work on independent GPUs. Shared snapshot requests are limited to
sixty seconds and reuse the anatomy decode at up to 1024px; older workers use
a bounded application-side frame-cache decode instead.
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
  failed/malformed batches try other healthy capacity, then bounded local fallback.
  A quarantined worker is not hammered with individual-frame retries.
- `GET /health`: liveness plus process-local statistics; never loads a model.
  `readiness` distinguishes uninitialized/ready/quarantined inference. HTTP 200
  alone is not readiness. Before first inference, device is `not loaded`.
- `POST /ready`: explicit inference smoke test on a generated neutral image.
  Does not reset a quarantined CUDA context.
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
higher saved concurrency never bypasses the per-worker admission guard. Batch
concurrency bounds requests, not individual frames; it does not qualify a GPU's
concurrent ONNX inference reliability.

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
CPU-only workers return 503 from the segment `/encode` endpoint; classification
and the software whole-video renderer below continue to work.

**Codec identity alone does not establish splice safety.** Vulgarr rejects
remote output when codec/profile/pixel format/resolution/color/frame-rate
metadata differ, when uploaded segment timestamp/duration drift is detected,
or when traced HEVC SPS/PPS derived-state fields differ or cannot be probed.
NVENC cannot reproduce every x265 knob, so HEVC offload may legitimately fall
back on an otherwise healthy worker. A returned MKV must decode cleanly before
it becomes a completed segment. The existing concatenation, whole-output decode
verification, matched-x265 CPU encoder and whole-file fallback remain intact.
Any transport/encode/validation failure retries the original local encoder.

## Whole-video recovery offload

A split/keyframe/concat failure can select whole-file recovery **before any
`/encode` request is made**. That used to bypass workers completely. Recovery
now submits a pollable `/render` job to `SPF_REMOTE_ENCODE_URL` too. Rebuild and
redeploy **both vulgarr and the workers**; an older worker returns HTTP 404 and
vulgarr safely uses server CPU, with a warning and explicit queue phase.

```dotenv
# Vulgarr container (use the actual address/port reachable from the worker):
SPF_REMOTE_ENCODE_URL=http://gpu-host:7861
SPF_INTERNAL_BASE_URL=http://vulgarr-host:8000
SPF_INTERNAL_TOKEN=<a-long-random-secret>

# Worker container; same scheme/host/port as SPF_INTERNAL_BASE_URL:
SPF_SOURCE_BASE_URL=http://vulgarr-host:8000
SPF_RENDER_WORK_DIR=/work
```

The worker streams the signed, file-scoped HTTP source; it needs neither a
shared media mount nor the signing secret. Source transport configuration is
required, but `SPF_REMOTE_EXTRACT` and `SPF_CLASSIFIER_URLS` are **not** required
for recovery rendering. Missing source settings are explained in the queue and
logs rather than silently selecting local encoding. Use private networking/TLS;
render capabilities last up to the 24-hour job deadline plus one hour so later
range reads do not expire mid-encode. Vulgarr redacts their query strings in
Uvicorn access logs; configure external proxies/log collectors to do likewise.
Do not log or share signed source URLs.

For H.264/HEVC, the worker tries NVENC with CUDA decode, then NVENC with CPU
decode, then codec-matched **software encoding on the worker**, not on the
vulgarr server. Other codecs use worker software encoders. Output codec,
resolution, decoded pixel format and duration are checked before accepting a
render. Whole-video NVENC is safe from the *mixed-SPS/PPS splice failure class*
because every video frame is encoded in one continuous session; this does not
relax any segment compatibility checks or replace real-media qualification.

- `POST /render`: validates the restricted blur graph/source origin and returns
  a job ID (202); `GET /render/{id}` reports queue state, backend and progress.
- `GET /render/{id}/output`: downloads the completed video-only MKV and removes
  its temporary files afterward. `DELETE /render/{id}` cancels ffmpeg and cleans
  up; worker shutdown cancels jobs too.
- One shared encode slot serializes `/encode` and whole-video renders. At most
  two render jobs/results are retained per process. A 24-hour deadline covers
  queueing and all attempts; unclaimed completed results expire on later
  submissions. Run **one uvicorn process per worker container**, not multiple
  `--workers` sharing the in-memory job registry.
- Provide enough worker disk space for full video outputs. The compose example
  gives each worker its own `/work` volume. `SPF_RENDER_WORK_DIR` must be dedicated
  to that process: startup sweeps its `vulgarr-render-*` directories left by hard
  kills. It never sweeps shared system `/tmp` or unrelated directories.
- `/stats` advertises `whole_video_render: true`, render activity and completed
  render count. The queue worker panel shows those separately from inference
  requests. Its device/providers field describes **inference**, not NVENC.
- The server still performs metadata probes, video validation, final audio/mux
  and atomic publication. Seeing server-side ffmpeg does not itself mean blur
  encoding is local. The queue explicitly says `Remote whole-file blur`,
  `Downloading`, `Validating`, or `server CPU (remote fallback)`.

Worker failures and invalid output preserve the local CPU safety net and do
not install a partial video checkpoint or overwrite the published edit. Network
loss/host crash cannot resume a whole-video encode in place; it may be retried.

### CUDA decode / green-half-screen fix

FFmpeg 4.4's CUDA-decode path can hand semiplanar NV12/P010 frames to `boxblur`.
On the deployed Ubuntu 22.04 build, this reproducibly zeroes the right half of
both chroma planes **only while blur is enabled**. The result is a gray/green
half-screen that can decode without any errors, so codec/duration/decode checks
alone do not detect it.

The worker explicitly converts to the original ffprobe pixel format **before**
all blur enable windows (for example `format=pix_fmts=yuv420p10le` for Main10).
GPU decoding and NVENC encoding remain enabled, without forcing 10-bit sources
through 8-bit conversion. New clients require the worker acknowledgement
`filter_layout: planar-v1`; older workers are cancelled and fall back locally
instead of publishing their visually corrupt output. `/stats` advertises
`render_filter_layout: planar-v1`, and the worker panel flags a missing fix.

Rebuild/redeploy both images and **Apply scenes again** to regenerate affected
edits, then reopen playback. Already-published files are not automatically
rewritten. The blur recipe fingerprint invalidates old intermediates, but does
not remove scan checkpoints or approved scene selections.

An opt-in synthetic regression exercises CUDA and CPU decode through NVENC for
8-bit H.264, 8-bit HEVC and 10-bit HEVC. It checks both chroma halves on every
frame, including blur boundaries, and asserts that bit depth is retained:

```bash
SPF_TEST_RENDER_CONTAINERS=vulgarr-worker-worker-0-1,vulgarr-worker-worker-1-1 \\
  python -m pytest tests/test_remote_render_gpu.py -q
```

It loads the repository's candidate renderer in an isolated diagnostic process
inside each worker. It does **not** modify/restart the running service or prove
that the deployed daemon has been upgraded; check `/stats` for that. Synthetic
fixtures supplement, not replace, the real-media checklist below.

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
5. Exercise whole-file recovery (including a source that fails keyframe/split
   planning). Confirm `/render` is queried, the queue identifies the worker
   backend, and the server does not run a boxblur/video encoder for that path.
   Repeat with NVENC unavailable to exercise worker CPU; interrupt a render and
   a result download to check cancellation/cleanup and local fallback. Check
   delayed audio, 10-bit/HDR metadata and scene on/off timing in the final mux.
6. If any mismatch/corruption appears, unset `SPF_REMOTE_ENCODE_URL`; do not
   relax the compatibility checks to make a particular GPU pass.
