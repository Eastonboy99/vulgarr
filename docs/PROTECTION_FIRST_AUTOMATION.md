# Protection-first automatic clean versions

## Scope and activation

Opt in under **Settings → Video Processing → Contextual nudity discovery →
Protection-first**. Keep the local LM Studio endpoint/model configured and select
clean movie/TV output folders. The application does not change Plex libraries,
move originals, download models or configure LM Studio's GPU placement.

The setting is off by default so upgrading does not silently approve scenes or
start expensive processing. Saving it does not enqueue a library-wide rescan.
For existing titles (including older review-only contextual candidates), start a
fresh ordinary scene scan once: old sparse coverage does not qualify for this
mode. Do not use **Clear scenes & restart** unless intentionally deleting old
review results. Future Sonarr/Radarr arrival webhooks automatically request video
analysis and audio preparation without requiring Vulgarr processing tags. Bazarr
subtitle-arrival events also prepare audio and request any missing visual analysis,
even when the legacy audio-trigger switches are off.
Existing subtitle acquisition/polling remains in effect; absent subtitles or a
failed audio job can delay the clean copy rather than publish unprepared audio.
Importing a library list alone is not a new-arrival processing trigger.

Protection-first implies local contextual discovery and automatic Apply, even
if the independent discovery/legacy auto-process checkboxes are off. It uses
that discovery endpoint/model and the existing per-category and audio controls.
Legacy whole-title semantic analysis is incompatible, and legacy candidate
verification is disabled while protection-first is on. No cloud fallback exists.

## Processing

1. Independently sample the entire original with NudeNet at **at most 0.25s**
   intervals, retaining raw class evidence. Incomplete/gappy results fail closed.
   Selected-label hits are not discarded merely for being isolated short hits.
2. Sample four contextual frames per window of **at most four seconds**.
3. Decode the whole title for shot changes (`scene > 0.30`) and inspect the first
   part of each detected shot, including short shots between regular samples.
4. Recheck uncertain windows once at the same timestamps with **1024px** samples
   instead of 768px. A recheck cannot veto an existing positive. Persist structured
   frame states and source window provenance, not invented confidence scores.
5. Check two-second intervals around candidates to conservatively extend their
   boundaries; never narrow an original positive. Existing render padding applies.
6. Automatically decide using content preferences. Positive matches are filtered;
   unresolved uncertainty uses the strongest configured action among the examined
   categories. If those categories are all allowed, uncertainty does not invent a
   blocking category. Uncertainty remains explicitly recorded, not relabeled as
   proven nudity. Audio defaults/explicit per-scene choices remain effective.
7. Wait for a valid clean-audio master and no active/failed/latest incomplete audio
   work. Keep a newly prepared audio master without publishing an audio-only copy
   if visual qualification is still pending. The dispatcher recovers missed
   scan/audio handoffs, including after a restart. It also requests missing/stale
   audio preparation once sidecar/cached subtitles are available, so a protected
   manual scan does not need a separate audio button. A missing clean folder for
   a media type blocks audio processing rather than falling back to editing the
   source in place. Terminal audio/render failures are not retried forever.
8. Render the effective interval union once using the configured output folder.
9. Validate the hidden staged result before replacing any published file.

Explicit human rejects, Allow actions, audio exceptions, manually added windows
and human-reviewed/adjusted rows survive protected rescans. Render windows and
output corrections exclude explicit human allow/reject intervals; explicit
unmute choices also subtract from automatic mute windows. These exceptions are
intentional user choices and can reduce protection. Unresolved human/legacy
pending choices hold publication instead of being silently overwritten.

## Finished-output checks and bounded correction

Existing decode, duration and AV-interleaving checks remain. Additional checks
compare stream types, audio sample rates/channel counts, and relative playback
stream starts when timestamps are available (250ms tolerance). Long blur windows
also get an end-boundary decode check.

Then independently scan the **rendered** video with NudeNet and the local VLM:

- Anatomy samples are offset by half the source sampling cadence.
- Contextual samples are shifted by half a frame cadence, not half a window
  (which would accidentally repeat the original timestamps).
- Include shot-start windows. Missing/incomplete evidence or failed requests are
  errors, never negative classifications.
- Expected uncertain/obscured frames wholly inside known blur coverage are not
  mistaken for unhandled exposure. Discernible positive content is still a
  failure, including inside an existing blur window.
- A residual blocked interval adds auditable automatic correction coverage.
  Already-covered residual content strengthens blur without duplicate rows.
- Retry rendering **once**, with expanded/padded coverage and blur at least
  radius 90 / power 8. Persistent residual content or check failure prevents
  publication. There is no unlimited correction loop.

Source fingerprints, scan generation, content/connection/sampling preferences,
render settings, manual exceptions and current decisions are fenced before
publication. Changing analysis criteria requires a fresh scan; stale evidence
is not automatically considered qualified. Original files, the previous visible
edit and successful clean-audio masters remain separate. No original-video
fallback is installed in the clean output folder. This is a publication safeguard
for Vulgarr's configured destination, **not** Plex library enforcement.

Lengthy checks report progress and remain cancellable. Output extraction fallback
knows it already owns the audio/blur gate, avoiding recursive-lock deadlocks.
Cancellation/failure cleans staged output; correction work stays inside the owning
job's private work directory. Output-check metadata records successful attempts
and actual final blur strength.

Disabling this mode does not undo prior approvals or delete existing edits.
Outside the mode, contextual discovery returns to the review-first workflow.

## Qualification and limitations

**Sampling and AI cannot guarantee that every unwanted frame is caught.** Shot
thresholds can miss small visual transitions, and local models can misclassify
clothing/coverage. Protection-first accepts unnecessary blurring to reduce risk;
failures can leave a title unavailable or keep the previous successful version.
A last successful file is not automatically retracted when preferences change.
Your separate Plex clean library and source-library access remain your setup.

A full-title contextual pass plus independent output analysis is expensive.
Higher-resolution rechecks and shot detection add work. CPU-only processing can
take many hours; do not assume real-time processing or qualified GPU throughput.

Implementation regression qualification:

- Standard suite: **581 passed, 12 skipped**.
- Both diagnostic media workers: **605 passed**.
- Offline Chrome at **1100px and 390px**: no control overlap, overflow or script
  errors; all 18 NudeNet controls remain independent of contextual policy.
- Real local FFmpeg test: a one-second synthetic clip's half-second shot change
  produces the expected short-shot window.
- These are software/media-integrity checks, **not broad model-accuracy proof**.

### Opt-in annotated benchmark

`scripts/qualify_protection.py` runs the actual scan pipeline in a temporary DB,
without changing application scene rows/settings or enqueueing jobs. Use original
clips representing contextual/exposed nudity, towels/sheets, ordinary clothing,
swimwear, male shirtlessness, low light and rapid cuts. Independently annotate the
actual intervals your preferences require filtering; do not substitute loose
whole-scene labels for frame-accurate exposure intervals.

```json
{"cases": [
  {"name": "contextual positive", "path": "/media/bench/contextual.mkv", "blocked": [[2.0, 5.5]]},
  {"name": "clothed control", "path": "/media/bench/clothed.mkv", "blocked": []}
]}
```

```sh
python scripts/qualify_protection.py --manifest cases.json --report report.json
# Optional full render/output checks in a private large workspace OUTSIDE libraries:
python scripts/qualify_protection.py --manifest cases.json --report report.json \
  --render --work-dir /large/private/qualification
```

Reports measure covered/missed/unnecessarily filtered **seconds**, duration recall
and precision, exact missed intervals and runtime. Any overlap is not enough to
claim that a nudity interval was fully filtered. Reports mark incomplete batches.
The optional rendered benchmark uses original audio outside visual mute intervals:
it qualifies visual filtering, not profanity detection. Reports must be separate
JSON files, never a source or the annotation manifest. Rendering workspaces cannot
be inside configured source/clean roots. Temporary credentials/evidence/rendered
media stay in the private workspace and are removed on normal completion/failure.

Broad independently annotated movie/episode recall, false-positive performance,
HDR playback and full-title GPU throughput remain to be qualified on real media.
The runner makes that measurement possible; it does not manufacture benchmark
results or start a full-library scan.
