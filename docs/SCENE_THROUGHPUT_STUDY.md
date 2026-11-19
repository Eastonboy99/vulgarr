# Scene throughput study: cancelled Breaking Dawn scan

## Scope

Read-only audit of title **3807**, scene job **32**, followed by bounded local
microbenchmarks. No full-title scan, render, publication, job cancellation,
deployment, restart, model download or Vulgarr setting change was performed.
The job was already cancelled before investigation. Dedicated LM Studio benchmark
instances were loaded from existing weights and unloaded afterward.

This is a performance investigation, **not accuracy qualification or an enabled
replacement detector**. Prototype scripts and private source frames remain in the
local temporary study workspace, outside the repository. The accompanying JSON
contains only numeric results and categorical observations, never image data,
source paths, signed URLs or credentials.

## Finding 1: the largest delay was a broken worker/retry cascade

Times below are UTC; LM Studio's local log clock was UTC-05:00.

| Event | Timestamp |
| --- | --- |
| Job started | 2026-11-16 17:27:27 |
| Initial full-title anatomy checkpoint finished | 2026-11-16 17:42:11 |
| Worker-1 CUDA misaligned-address error recorded | 2026-11-16 17:44:33 |
| Late anatomy-refinement fallback warnings | 2026-11-17 02:41:29 |
| First contextual chat request | 2026-11-17 02:56:04 |
| Last completed contextual checkpoint record | 2026-11-17 04:38:16 |
| User-cancelled job finished | 2026-11-17 04:38:21 |

The initial anatomy pass took approximately **14m44s**. Approximately **9h14m**
then elapsed before the first contextual request. Most of that interval contains
repeated extraction/classifier failures during candidate refinement; the final
pre-VLM interval also includes shot discovery and extraction overhead. The logs
do not justify assigning every second of that interval to one operation.

Worker 1 recorded:

- **416** `POST /detect-batch` responses with HTTP 500.
- **4,602** `POST /detect` responses with HTTP 500.
- **17** `POST /scan` responses with HTTP 503.
- Repeated ONNX Runtime **CUDA failure 716: misaligned address**, followed by
  persistent CUDA/cuBLAS errors.

The app logged **47** remote-extraction fallbacks and hundreds of batch fallbacks.
It continued selecting the failed worker and retrying individual frames, rather
than quarantining it. This is the dominant actionable problem, not proof that
VLM inference alone consumed eleven hours.

The original cause of the CUDA fault is **not established**. GPU inference
concurrency, runtime/driver compatibility and device isolation require isolated
qualification. Do not assume that ordinary parallel FFmpeg render tests establish
concurrent ONNX inference reliability. Both live worker health endpoints could
return HTTP 200 without a loaded detector: reachability is not inference readiness.

Other observations:

- No overlapping Vulgarr audio jobs were found in the audited interval.
- The app container had no configured CPU quota or memory limit.
- Current scan settings yielded **61** coarse anatomy candidates, spanning
  **749 seconds**, followed by dense sampling at **0.1-second intervals** and
  configured 5-second/maximum 30-second padding. Full-span iterative refinement
  repeats work already represented in the coarse raw checkpoint.

## Finding 2: whole-title generative discovery is still expensive

Contextual discovery completed **507** windows in approximately **1h42m** before
cancellation. The final completed base window ended around **1,376 seconds**
(**22m56s**) into the movie. This had **not reached** the known missed example
around 36m54s. Extra shot windows make window count larger than elapsed footage
would suggest from four-second base windows alone.

The checkpoint contained two positive and 78 uncertain windows. These are model
classifications, not distinct scenes or ground truth. Uncertainty and boundary
rechecks had not yet completed, and no rendered-output pass had started.

The 507 retained language-stage timing records had a median near **10.08 seconds**
and total near **1.43 hours**. These internal timings do not include all external
extraction, request and orchestration time. Historical GPU offload placement was
not recoverable from the retained logs; it must not be inferred from token speed.

The displayed remaining-time estimate mixed elapsed work and overall progress.
It is not a reliable measurement of the remaining VLM phase, particularly after
hours of failed-worker retries.

## Bounded inference benchmark

Existing **Qwen3.5-4B Q4_K_M**, explicitly requested at full GPU offload, context
8,192, parallelism one, reasoning disabled. Loading increased RTX 3060 memory
usage by approximately 4.5 GiB; no other model instance was loaded. The instance
was assigned a separate benchmark alias and unloaded after each benchmark.

All requests went to loopback LM Studio with environment proxies and redirects
disabled. Four-frame contact sheets used a 2x2 layout, about 768x320 pixels for
this source. Separate low-resolution frames were approximately 384x160 pixels.

Five challenging windows used existing private pilot frames from the same
one-minute bedroom-region clip. Six additional normal-scene controls reused
historical frames in memory from the cancelled scan's local LM Studio logs.
The control indices were selected because their saved observations were all
clothed/no-people, then rechecked with the current detailed classifier.

**These controls are model-labeled references, not independent human labels.**
They are narrow, same-title probes and cannot establish recall or a library-wide
false-positive rate.

| Variant | Windows | Median request wall time |
| --- | ---: | ---: |
| Current detailed response, four separate 768px images | 11 | **4.609s** |
| Compact clear/candidate/uncertain response, four 768px images | 5 | **1.156s** |
| Compact response, four 384px images | 11 | **0.609s** |
| Compact response, one four-frame 768px contact sheet | 11 | **0.531s** |

There were **38** local inference requests total. Timings include the local HTTP
request/response, but exclude image preparation and NAS extraction. Detailed
responses generated roughly 260 tokens per window; compact responses generated
12-13. Fresh JPEG encodings were used across variants to avoid reusing identical
image-request bytes. Prompt/image caching and a warm single model mean these
numbers remain microbenchmarks, not sustained full-title throughput guarantees.

### Decision trade-off

- Both low-resolution compact variants flagged the known contextual example.
- The current detailed classifier returned clear on all six normal controls.
- Both compact variants returned clear on **four of six** normal controls and
  escalated the other two as candidate/uncertain.
- The compact contact sheet also escalated a towel-region control that the
  detailed classifier considered clear.

Therefore **do not auto-filter solely from the compact screen**, and do not call
the difference a calibrated accuracy result. On this tiny biased set, additional
fine classification would still be needed for several controls. A fast front
screen is not an 8.7x whole-pipeline improvement when many windows escalate to
expensive processing. Conversely, reducing output prose is a measured opportunity
that does not require new model weights.

## Bounded extraction benchmark

The original source is **1920x800, HEVC 10-bit, 24000/1001 fps**. Temporary JPEGs
were produced on the NAS and deleted; they were not exported.

| Extraction experiment | Images | Wall time |
| --- | ---: | ---: |
| Four independent seeks near the known example | 4 | **2.885s** |
| One decode over an eight-second region, sampling every two seconds | 4 | **2.132s** |
| One decode over a sixty-second region, sampling once per second | 60 | **6.847s** |

Different FFmpeg sampling operations do not necessarily select identical source
frames; these are extraction-throughput comparisons, not pixel-equivalence tests.
The one-minute JPEG output was approximately **0.82 MB**. Cache size extrapolates
to roughly 102 MB for this title at the measured content-dependent compression.
Do not treat that as a worst-case storage bound.

At subsecond model latency, repeated process startup and seeking dominate.
Prefer emitting a bounded, timestamped contextual frame cache from the **same
continuous decode** that already produces anatomy samples, rather than another
whole-title decode or four new seeks per window. Preserve source/configuration
fingerprints, cancellation, actual sample provenance and disk limits.

## Recommended implementation order

1. **Worker reliability and bounded failure handling first.** Quarantine a worker
   after CUDA/session failures; route to independently healthy capacity. Add a
   bounded inference smoke test/readiness state, per-worker inference admission,
   explicit retry/CPU-fallback budgets and stage-specific telemetry. Investigate
   the original fault before returning that worker to service. Budget exhaustion
   withholds a new result and retains the previous published edit; it never means
   clear/no content. No worker was restarted during this investigation.
2. **Reuse existing evidence and decoding.** Keep the independent 4-fps anatomy
   pass. Use its raw evidence for routine candidate decisions; reserve denser
   rescans for bounded boundary/uncertainty regions instead of repeatedly scanning
   every full candidate. Share decoding with a private contextual frame cache and
   shot observations where feasible. Requalify sampling semantics rather than
   silently pretending changed timestamps are the old evidence.
3. **Prototype compact classification and bounded escalation.** Retain independent
   contextual discovery even when NudeNet has no hits. Avoid generating four
   explanations and a summary for every clearly ordinary window. Do not replace
   policy-aware evidence with a generic NSFW boolean or invent calibrated scores.
   Preserve strict frame identity/structured evidence and the existing defenses
   against instructions embedded in images. The screening prototype is not a
   production-hardened replacement. Validate compact per-frame states/triage on
   an independently annotated corpus.
4. **Avoid a second exhaustive generative output pass by default.** Keep full
   mechanical media/AV checks, and measure independent inexpensive visual checks
   plus targeted semantic verification of affected regions. This changes the
   assurance trade-off and requires explicit qualification, not merely faster
   configuration values.
5. **Qualify one full title before enabling a library rollout.** Report stage wall
   times, active/wait/retry time, actual model/device, call counts, escalation rate,
   misses, extra filtered seconds, output integrity and source protection.

A 30-60 minute analysis target is **not yet demonstrated**. Even the compact
screen alone projects to about **16.5 minutes** for the 1,864 base windows, or
about **24 minutes** if the early observed shot-window density continued. Those
exclude extraction, escalations, anatomy analysis, output checks and rendering.
The latter window count is an extrapolation, not a recovered whole-title plan.

Even 30-60 minutes per movie means roughly **21-42 continuous processing days for
1,000 movies** on one serial pipeline, before extra rendering overhead. A truly
large back catalog may need a fast dedicated visual classifier/image encoder,
parallel capacity, or both. No suitable pretrained CLIP/SigLIP vision weights were
found in the inspected Hugging Face cache, and none were downloaded. That option
remains untested and needs separate acquisition/qualification authorization.

## Unchanged safeguards

Keep atomic scan activation, explicit manual exceptions, current-policy/generation
fencing, reliable audio masters, original/last-successful-file preservation and
configured-output-folder publication gating. No Plex library enforcement is
introduced. No prototype screening result was approved, activated or published.
