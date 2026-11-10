# Scene reset, audio and content-policy implementation

> **Historical initial-rollout report.** The follow-up [NudeNet content-preferences design](NUDENET_CONTENT_PREFERENCES.md) supersedes the menu/taxonomy, default scan UI and policy-schema descriptions below. The current menu exposes NudeNet's exact 18 body-part labels, not the initial semantic/disabled category list. Audio, atomic-scan and publication safeguards described here remain in place.

## Shipped behavior

- **Process/Reprocess Video**, **Both**, single-title, season and show scan actions create a fresh job generation. All previous scene windows/review decisions, including approved and manual windows, are replaced **only after successful analysis**. The standalone scan form offers **Keep manual scenes**. Audio-only actions retain scene decisions.
- Database conditional writes claim scans and exclude other active scene jobs. Dispatch also claims queued jobs atomically. Activation checks the requested generation and job state. Completion is part of the activation transaction; an already completed generation cannot replay over human review. Restart resumes the same job's checkpoint. Prior-generation individual forms cannot hit newly generated random scene IDs; bulk forms carry a generation fence.
- Previous results remain visible, explicitly labeled as previous during scanning. A failed/cancelled scan retains them and the published edit. No partial candidates are published by progress commits.
- Global visual policy is schema version 1, with stable IDs and `allow`, `blur`, `blur_mute` actions. Multiple labels are combined: allowing a neckline does not cancel a bikini match. There is no bathroom/bedroom/beach/pool blocking category.
- Global audio defaults to muting enabled blurred scenes. Nullable audio overrides distinguish inheritance from explicit sound/mute choices. Migration preserves **both** old Boolean values as explicit overrides. **Use global audio defaults** deliberately clears those overrides; it does not regenerate media.
- Every renderer entry point resolves the same effective action. All real playback audio tracks are continuously decoded, muted and encoded as AAC-LC if scene muting is needed. No copied AAC/HE-AAC packets are concatenated with incompatible locally encoded packets. No forced stereo downmix or sample-rate change is used. With no mute windows, audio is copied.
- Edits rebuild from original video and the validated un-scene-muted clean audio master. Empty effective sets copy pristine video, allowing explicit regeneration to remove previous blur/mute windows without another video encode. The existing two-pass AV interleaving/sparse-stream finalizer is unchanged.
- Applied policy and effective scene/render-setting revisions are persisted independently from detection-cache identity. Review previews recalculate stored evidence without model calls. Settings save never starts a library-wide scan/render. Changed policy, scene actions or render settings enable explicit per-title regeneration. A final pre-publication fence rejects stale rendering and reserves the DB writer through publication/commit.

## Classifier capabilities and opt-in

Local NudeNet remains a legacy exposure-candidate backend. Its covered-anatomy labels **do not identify bikinis or necklines**. Raw detections are now retained in v2 checkpoints and per-scene detector evidence. Updated remote workers advertise raw-detection schema/capability; older scalar-only responses are still accepted but do not acquire invented semantic evidence or coverage. App and worker NudeNet dependencies remain pinned to 3.4.2; the installed local detector version participates in the resume fingerprint.

Enable **whole-title clothing/nudity analysis** in Settings to use the existing configured OpenAI-compatible vision endpoint/model. This is independent of the old yes/no-verifier switch. Explicitly opting in permits sampled private frames to leave the server and may incur provider charges. The new backend:

1. Proposes every temporal window, not just NudeNet hits.
2. Samples three representative frames per window, sequentially (bounded concurrency 1).
3. Requires strict JSON with multiple category confidences, explicit uncertainty and a visual explanation.
4. Keeps policy-independent structured evidence, configured model identity, schema and coverage. Allowed observations remain stored for later policy changes.
5. Resumes only a source/model/endpoint/sampling-compatible contiguous window prefix. A corrupt tail is truncated. A provider/extraction/validation failure aborts the result swap, preserving the previous scene set and media.
6. Records successful whole-title sampling coverage, including titles with no matched categories. Coverage means categories examined at the sampled frames, **not exhaustive safety**.
7. Uses structured classification on re-verification when semantic analysis is enabled; structured clothing evidence is never sent back through the old swimwear-allowing yes/no prompt. External re-verification is rejected without an explicit analysis opt-in.

At the default 10-second window, a two-hour title requires about **720 calls / 2,160 images**. Cost and latency depend on the configured provider; there is no paid service dependency when disabled. Three stills can miss short shots and cannot qualify temporal sexual acts or violence. Provider/model names are configured provenance, not a guarantee of a provider's immutable model revision.

### Available visual labels

Exposed breasts/genitalia/buttocks; ordinary low-cut neckline; covered chest outline; bikini/two-piece swimwear; one-piece swimwear; underwear/lingerie; male tight underwear.

The default profile allows low-cut necklines and covered chest outlines and blocks bikini/selected nudity/underwear with audio mute. Location alone never blocks a scene. Manual action overrides take priority; approval alone cannot turn an allowed category into a blocked action.

### Deliberately unavailable

The new policy UI disables sexual-activity/implied-activity/gesture, violence, disturbing/self-harm, transcript/OCR self-harm-reference and medical categories. There is no qualified temporal/transcript backend here. The legacy optional verifier still has its existing sexual-content verdict, but that does **not** establish structured temporal-category coverage. Existing profanity word/severity/exclusion processing remains separate and operational. No VidAngel annotations are scraped and no coverage parity is claimed.

## Qualification

Commands run locally:

```bash
python -m pytest tests/ -q
# 425 passed, 12 skipped, 5 dependency warnings

SPF_TEST_RENDER_CONTAINERS=vulgarr-worker-worker-0-1,vulgarr-worker-worker-1-1 python -m pytest tests/ -q
# 449 passed, 5 dependency warnings

git diff --check
# clean
```

New regression coverage includes real SQLite generation activation, repeated fresh scans, approved/manual replacement, manual preservation, obsolete/cancelled/replayed generations, concurrent queue claims, legacy Boolean migration, stale Apply rejection, bulk-form fencing, policy union/overrides/validation, settings export/import, no-consent rejection, broad semantic proposals independent of NudeNet, strict provider responses, contiguous semantic checkpoint recovery, raw-evidence retention and empty-set restoration.

Real audio diagnostics execute the **candidate continuous-filter function** in isolated worker processes with synthetic non-silent AAC-LC stereo and 7.1 media. Each fixture has two playback tracks, one delayed by 350 ms. Both workers pass normal/overlapping windows, a window starting at zero and a window ending at EOF. Decoded audio stays audible outside the window and below 0.0001 RMS (-80 dBFS) in its interior; channel count/layout, 48 kHz sample rate and track delay are checked. Tests are not daemon upgrades and do not touch user media. Existing opt-in interleaving/stream-preservation and GPU/10-bit synthetic regressions also pass.

### Not yet qualified / deployment gates

- **Live clothing/nudity model accuracy is not qualified.** Example-category/provider tests use explicitly mocked observations. Supply licensed/user-authorized low-cut-shirt, bikini and bathroom-dialogue fixtures before measuring real false positives/negatives, recall, provider latency and cost. No specific user's scene contents have been inferred or fabricated.
- **HE-AAC source decoding and regeneration are not real-media qualified.** Continuous AAC-LC avoids configuration splices structurally, but the diagnostic workers have no HE-AAC encoder and no authorized HE-AAC fixture was supplied. Do not claim that acceptance criterion passed.
- No affected movie/episode has been regenerated or checked in Plex Web/Direct Play. Synthetic 10-bit tests do not establish real HDR-title qualification.
- Temporal/transcript/violence/medical categories remain staged work, not working toggles.
- Sampling interval controls cost/recall. Window-level evidence and confidence do not provide frame-exact clothing boundaries; manual review/padding remains important.
- Re-verifying existing windows cannot discover content outside those windows. Use a fresh whole-title semantic scan when category coverage is absent.

## Deployment/regeneration checklist

1. Back up the DB/settings and retain original sources and last successful edits. Review the Boolean migration behavior and new fresh-reset warning.
2. Deploy app/worker changes through the normal operator-approved process. Worker v2 raw evidence is optional for compatibility; it is not a semantic classification service.
3. Configure a suitable vision-capable OpenAI-compatible endpoint/model only if wanted; explicitly opt in to external frame analysis. Validate provider privacy/cost terms.
4. Qualify the authorized visual and HE-AAC fixtures above. Inspect category explanations, uncertainty, audio overrides and all selectable playback tracks.
5. Save the desired policy. For legacy/missing category coverage, explicitly scan the intended title. Otherwise review the recalculated effective actions without model calls. Reset legacy audio overrides only if that is the desired change.
6. Explicitly regenerate the intended movie and episode. Verify silence, restoration of newly allowed windows, AV sync, subtitles/attachments/chapters, source preservation and Plex Web/Direct Play. Include actual HDR media before claiming HDR qualification.
7. Expand to more titles only after these checks. Settings save alone never rewrites a library.

No new Python dependency was added. Nothing was pushed, deployed or restarted; existing user media remains unchanged. The original untracked plan and `graphify-out/` are not included in implementation commits.
