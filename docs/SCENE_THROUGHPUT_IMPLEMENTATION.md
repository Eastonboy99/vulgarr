# Bounded scene throughput controls (protection-v3)

Implements the reliability/shared-frame recommendations from
[the measured study](SCENE_THROUGHPUT_STUDY.md). **This is not a demonstrated
30–60 minute full-title pipeline or an accuracy-qualified fast detector.**

## Runtime changes

### Classifier admission and failure budgets

- The client admits one request per worker URL; independent workers can run in
  parallel. Quarantine is shared across requests/titles using the HTTP client.
- A failed transport/status/schema attempt quarantines that URL for five minutes.
  Waiting requests recheck quarantine before sending. Work tries other healthy
  configured workers before considering CPU fallback. A newer worker's explicit
  source/decode failure instead disables only that extraction route for the
  current title; healthy inference/other titles are not globally quarantined.
- A title scan shares a budget across coarse anatomy sampling and all candidate
  edge checks: **12 failed remote attempts**, **64 local fallback frames**, and
  **30 cumulative seconds of CPU fallback**. The output anatomy pass also gets a
  bounded scan budget. Configured-remote outages no longer silently turn into
  hours of whole-title CPU fallback. Intentional CPU-only installations, with no
  classifier URLs, keep the normal local path.
- `/scan` client requests time out after 90 seconds rather than 660. Worker decode
  subprocess attempts time out after 90 seconds rather than 600. A CPU retry can
  still follow a failed CUDA decode; neither attempts nor retries mean clear.
  Configured-remote local extraction also has a 120–1800 second timeout, scaled
  to the bounded region; intentional CPU-only extraction keeps its existing path.
- Budget exhaustion and extraction/frame-analysis failure abort activation;
  partial score JSON can remain resumable, but is never a successful generation.
- Cancellation joins extraction/classification tasks before deleting their
  temporary directories. A timed-out native CPU thread cannot be forcibly
  killed: the budget stops admitting more calls and rejects its eventual result.

The worker serializes detector construction/inference across `/scan`, `/detect`,
`/detect-batch`, and readiness smoke tests. Provider/CUDA failures quarantine the
process until operator recovery. An inference-lock admission wait is bounded at
30 seconds; a timed-out waiter also quarantines the process. Even if an older
native call eventually returns, that return cannot clear quarantine.

`GET /health`/`GET /stats` never load a model. They distinguish `uninitialized`,
`ready`, and `quarantined`, report inference readiness/failure counts, and expose
only a sanitized error kind/CUDA numeric code. `POST /ready` performs a real
inference on a generated neutral 64x64 BMP. The client uses it when a newer
worker reports uninitialized inference. Legacy workers remain supported; their
first real response must still validate. The client does not use environment
proxies or follow redirects.

**Serialization is a defensive control, not proof that the original CUDA 716
root cause has been fixed.** Qualify real ONNX inference on each physical GPU
before returning a failed runtime to service. No automatic container/GPU reset
is introduced.

### Shared private snapshots

Protected source scans request `include_samples` on the existing `/scan` RPC.
A compatible worker emits snapshots at t=0.5, 1.5, 2.5, ... seconds (one per
second) from the frames already decoded for anatomy analysis. They have their own `fps-cache-v1` protocol;
raw NudeNet score schema/version and the 18 policy labels are unchanged.

Snapshots retain up to **1024px** so discovery at 768px and a 1024px uncertainty
recheck use the **same actual decoded frame**. A recheck never enlarges a 768px
thumbnail and pretends it contains more detail. The worker uses its existing
OpenCV dependency. The app uses explicitly declared Pillow.

- JPEG payload storage is bounded to **256 MiB per active title**, with LRU
  eviction, plus small checksum sidecars and temporary decode files.
- Responses are bounded to 60-second shared-frame requests. Only expected,
  unique timestamps and bounded, decodable JPEGs are adopted. A bad cache
  response cannot poison the score checkpoint.
- Cache identity binds the source scan fingerprint and protocol. SHA-256 sidecars
  reject corrupt files. Source/configuration changes still fence publication.
- For older workers or resumed raw-score checkpoints lacking images, missing
  grid frames trigger **one continuous, bounded chunk decode**, not four seeks
  per base window. A warmed chunk with a legitimate EOF miss falls back to exact
  seeking; it never becomes a negative observation or an infinite decode loop.
- Off-grid shot/boundary requests keep exact seeking. Default four-second base
  windows benefit fully; shorter/non-grid configured windows may still seek.
- FPS-selected pictures can differ from an exact seek by up to half the protected
  0.25-second anatomy interval plus source-frame quantization. Backend evidence
  records `sample_sources` for each frame. Checkpoint identities and protection
  qualification include the changed sampling protocol; old coverage is not
  silently treated as equivalent.
- A fallback cache decode has a 120-second deadline and cancellation kills its
  FFmpeg child. Mandatory shot decoding has a maximum 30-minute wait; a timeout
  aborts rather than silently removing shot checks.
- No rendered-output check can read the original's frame cache. It still analyzes
  the actual rendered file with its independent sampling offset.
- Successful scans already remove their work directory. Failure/cancellation also
  discards private JPEG caches while retaining resumable JSON evidence. Abrupt
  process termination may retain cache files until job recovery/maintenance.

### Bounded refinement and telemetry

Full-title anatomy coverage remains at least four samples/second. Protected
candidates reuse those raw observations. Dense refinement samples only two
four-second edge regions (merged if overlapping), never the entire candidate
interior or successively larger whole spans. It never shrinks the coarse scene,
retains conservative padding, and unions duplicate-timestamp raw observations
so a later negative cannot veto an earlier positive. Terminal decisions continue
through any longer container/audio tail.

Independent full-title contextual discovery and all shot checks remain mandatory.
Optional high-resolution uncertainty rechecks are capped at **64**; contextual
boundary checks at **128**. Unrechecked uncertainty remains conservative blocking
evidence, not clear. Skipped boundary checks retain original conservative padding.
Endpoint/structured-output failures still abort, rather than being counted as
successful budget skips. Coverage metadata records completed/skipped counts and
cache hits/fallback decodes. These are fixed protocol budgets, not new hidden
live-setting changes.

Progress includes quarantine, failure and CPU-fallback counters and elapsed time.
The scan queue no longer extrapolates one overall ETA across unlike stages and
hours of retries. Rendering ETA behavior is unchanged.

## Experimental compact qualification (not a production switch)

`app/vision/compact.py` requests the existing eight policy-aware per-frame states,
without generated evidence sentences or a window summary. It preserves frame IDs,
uncertainty, strict structured output and the instruction to ignore text embedded
in images. It is **not** a generic NSFW boolean. It reuses the local-only,
DNS-pinned endpoint validator; there is no remote fallback or download code.

Its results deliberately cannot pass the application's publication-evidence
validator. There is no settings checkbox, automatic approval or publication path.
The tiny three-way-gate study did not qualify this different state-only protocol.

Run the tool from a repository checkout with the root requirements installed
(the scripts directory is not bundled in the ordinary app image). Use an existing
loaded local model and a private JSON manifest:

```json
{"cases":[{"id":"clothed_control","frames":["f0.jpg","f1.jpg","f2.jpg","f3.jpg"],
"annotation":"human","states":["clothed","clothed","clothed","clothed"]}]}
```

```bash
python scripts/qualify_compact.py --manifest /private/cases.json \
  --model EXISTING_LOCAL_MODEL_ID --output /private/new-report.json --edge 768
```

There are at most **32 cases / 64 requests**, with a default **300-second** wall
budget (maximum 600). `--performance-only` permits unlabeled/model-labeled inputs
but produces **no accuracy metrics**. Ordinary comparisons require explicit human
per-frame annotations and report missed-positive/extra-flagged frame counts and
state disagreements; they are not calibrated confidence or temporal recall.
Requests alternate detailed/compact ordering across cases; warm vision/KV caches
may still affect timings. Reports contain states/timing/input hashes, not images,
source paths or API credentials. Existing output files are never overwritten.
All reports explicitly remain **not accuracy-qualified / publication disallowed**.

Before adoption, independently annotate diverse positives and ordinary controls,
including the known missed implied-nudity example, short shots, swimwear, towels,
male shirtlessness, low light, image-text attacks and HDR. Then qualify temporal
boundaries, a whole title, rendered output and actual GPU throughput. One-time
benchmark annotation is distinct from routine per-title review.

## Deliberately unchanged

- The detailed classifier remains the production default; compact speed/accuracy
  and end-to-end throughput are not established by unit tests.
- Full-title shot decoding and exhaustive rendered-output VLM verification remain
  in place. Replacing either with selective/cheaper checks needs qualification;
  silently removing those independent opportunities to catch a miss is not a
  performance fix.
- Atomic scene activation, originals, last successful edits/audio masters, manual
  exceptions, audio/mux safeguards and generation/policy/source publication fencing.
- Configured output folders and Plex library management.

## Deployment and qualification

Both app and worker **images need rebuilding** from the updated source. The app
has a new explicit Pillow dependency and workers must include `inference.py`.
Installing code alone into an older app image fails early rather than disabling
healthy workers later. An updated app can use old workers with client admission,
quarantine and chunk-cache fallback; worker-side CUDA isolation/shared snapshots
require the rebuilt workers.

Protection qualification is now **protection-v3**. After an authorized deployment,
use an ordinary **Scan** to establish fresh qualification. Do not use Clear &
Restart merely to upgrade; the last successful edit remains intact on failure.
No deployment, restart, live setting change, movie/full-library scan or model
acquisition is part of this implementation.

Tests include fake-runtime concurrency/CUDA-failure injection, admission/fallback
budgets, no late checkpoint writes after abort, real SQLite/file retention,
cache integrity/eviction/EOF/cancellation, strict experimental response parsing,
and real synthetic FFmpeg decode/shared-snapshot tests on both existing workers.
The FFmpeg tests use isolated processes and fake detectors, not a claim of real
ONNX/CUDA runtime qualification. Existing AV/render integrity regressions remain
part of the full suite.

Validation at implementation: **643 passed / 14 opt-in skips** in the standard
suite; **671 passed** with both existing worker containers enabled. The tests do
not establish model recall, resolve the original CUDA fault, or prove a full-title
runtime target.
