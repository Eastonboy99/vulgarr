<p align="center">
  <img src="app/static/favicon.svg" width="72" alt="Vulgarr">
</p>

<h1 align="center">Vulgarr</h1>
<p align="center">A self-hosted, Radarr/Sonarr-styled profanity &amp; nudity filter for your Plex/Jellyfin library.</p>

<p align="center">
  <a href="https://github.com/dpresti/vulgarr/actions/workflows/tests.yml"><img src="https://github.com/dpresti/vulgarr/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/dpresti/vulgarr/pkgs/container/vulgarr"><img src="https://img.shields.io/badge/ghcr.io-dpresti%2Fvulgarr-blue?logo=docker" alt="GHCR"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/dpresti/vulgarr" alt="License"></a>
</p>

<!-- GHCR has no public API for live pull-count/image-size stats the way Docker
     Hub does, so those two can't be added here as live badges without also
     publishing to Docker Hub. -->

Vulgarr runs two independent pipelines against your library:

- **Audio**: adds a second, selectable **"Clean"** audio track to movies and
  episodes by muting the audio under profane subtitle cues. Detection is
  subtitle-based (matches an `.srt` against a configurable, severity-tagged
  word list) — never ASR/transcription. The original audio track's bytes are
  never touched; a verified copy with the new track atomically replaces the
  file on disk, and the original is always kept as a timestamped backup.
- **Video**: scans for candidate nudity/sex scenes using a local frame
  classifier, queues them for a quick manual approve/reject in the UI (or
  auto-approves, if you turn that on), then produces a separate **"Vulgarr
  Edit"** copy of the file with those scenes blurred and optionally muted —
  the original file is never modified. Optional independent **local contextual
  nudity discovery** can find unclothed people even when intimate anatomy is
  obscured. It samples the whole video through a local vision model (such as
  LM Studio), not just NudeNet hits, and always requires human review.

Both pipelines are entirely opt-in per title/season/show and run
independently of each other. Built to sit alongside an existing
Sonarr/Radarr/Bazarr/Plex/Jellyfin stack and stay out of the way: it watches
for new imports, does its thing, and the results just show up as extra
selectable tracks/files your player already knows how to handle.

## Screenshots

| Library | Settings |
|---|---|
| ![Library](docs/screenshots/library.jpg) | ![Settings](docs/screenshots/settings.jpg) |

| Activity queue | Word List |
|---|---|
| ![Activity queue](docs/screenshots/queue.jpg) | ![Word list](docs/screenshots/wordlist.jpg) |

## How the audio pipeline works

1. **Parse** the episode/movie's `.srt` (`app/subtitles/parser.py`) into timed cues, stripping HTML/ASS-style tags and normalizing line breaks.
2. **Match** each cue's text against the word list (`app/subtitles/matcher.py`) — case/punctuation-insensitive, whole-word by default, severity-tagged (Child/Teen).
3. **Build mute intervals** from matched cues (`app/audio/mute.py`) with small padding and merging of close cues.
4. **Remux** (`app/mux/remux.py`): ffmpeg stream-copies every existing stream (video, original audio, subtitles, chapters) untouched and appends one new AAC audio track with a `volume=0` filter gated to the mute intervals, tagged `title=Clean`. The result is verified (stream count, duration, and a real decode of the new track) in a temp file before anything touches the original. Only then is the original moved to `/data/backups/<relative path>.<timestamp>.orig` and the verified file atomically swapped into place.

Because this is stream-copy plus one small filtered audio track, most of a
file's bytes are literally copied, not re-encoded — fast, and no quality
loss on video/original audio.

## How the video pipeline works

Entirely separate feature, entirely manual to start (never runs on import —
only from an explicit "Scan for scenes" click, since a scan has a real CPU
cost and Apply produces a full second encoded copy of the file):

1. **Scan** (`app/vision/classifier.py`): samples frames from the video at a
   configurable interval and classifies each with a self-hosted model
   (local by default, or optional remote GPU workers) — no ASR/subtitles
   involved, this is purely a visual classifier. Local extraction overlaps
   classification across bounded chunks, retaining per-frame resume checkpoints.
2. **Cluster** (`app/vision/scene_cluster.py`): merges nearby hits into
   candidate scenes, then does a second, denser re-scan of just each
   candidate's padded window to refine its actual start/end and compute a
   confidence stat used for one-click bulk-approval later.
3. **Contextual discovery** (`app/vision/contextual.py`, optional): independently
   samples four ordered frames per window across the whole title through a
   private/local vision endpoint. Obscured-nudity and uncertain candidates are
   combined with detector results without duplicate overlaps. No public
   endpoint, redirect, environment proxy or remote fallback is allowed.
4. **Review**: approve, reject or adjust candidates by hand. Optional auto-approval
   applies to eligible NudeNet results, never contextual AI candidates. Pending
   review blocks automatic publication. The separate candidate-verification
   setting only rechecks existing legacy candidates; it cannot find missed
   scenes or veto exact NudeNet/contextual evidence. Its configured endpoint
   determines privacy/cost: LM Studio is local; a cloud provider is not.
5. **Apply** (`app/mux/scene_blur.py`, `app/scenes/pipeline.py`): re-encodes
   just the approved windows with a boxblur (adjustable intensity, with a live
   preview in Settings) and optionally mutes their audio too, producing a
   sibling **"Vulgarr Edit"** file — the original is never touched. Re-running
   Apply after approving/rejecting more scenes always rebuilds from whatever's
   currently approved.

## Separate libraries for edited videos

In **Settings → Edited Video Output**, set **Movie output folder** and/or
**TV output folder**. They are independent absolute paths **inside Vulgarr's
container**, not host paths. A blank setting keeps the existing behavior for
that media type: audio processing updates the source with a backup/swap;
scene Apply creates a sibling ` - Vulgarr Edit` file.

When configured, **both audio and scene edits leave the source video as-is**
and publish the edited copy into that type's destination. This does not undo
in-place edits performed before the setting was enabled. Paths are mirrored
relative to `SPF_MEDIA_ROOT`, retaining movie/show/season folders and adding
` - Vulgarr Edit` to the filename. For example, with `SPF_MEDIA_ROOT=/media`:

| Source | Output setting | Edited video |
| --- | --- | --- |
| `/media/Movies/Film (2024)/Film.mkv` | `/edits/movies` | `/edits/movies/Movies/Film (2024)/Film - Vulgarr Edit.mkv` |
| `/media/TV/Show/Season 01/Show - S01E01.mkv` | `/edits/tv` | `/edits/tv/TV/Show/Season 01/Show - S01E01 - Vulgarr Edit.mkv` |

Point Plex at the **mirrored movie/TV subtree**, e.g. `Movies` or `TV` in those
examples, using the path visible to Plex. Sources on additional mounts outside
`SPF_MEDIA_ROOT` go under `_external/<absolute source hierarchy>` rather than
being flattened into colliding basenames. Changing or clearing a setting never
moves/deletes previous edits; process again or Apply to publish at the newly
selected location. Embedded streams/metadata are retained; external subtitle,
poster and metadata sidecar files are not copied.

Add writable destination mounts to your deployed fork's compose service, e.g.:

```yaml
services:
  vulgarr:
    volumes:
      - /srv/vulgarr-movies:/edits/movies
      - /srv/vulgarr-tv:/edits/tv
```

Mount those same host directories into Plex as needed. Vulgarr creates missing
subdirectories at processing time; its configured PUID/PGID needs write access.
A destination/write/copy error fails the job without silently editing the
original instead. Cross-filesystem publication copies into a hidden temporary
file on the destination, then renames it, so an incomplete copy never replaces
a previous successful edit.

### Combining audio and scene edits

Separate-output audio processing retains an **audio-only master** under
`<edited-video-folder>/.vulgarr/clean_audio/<title-id>/`, containing original
audio plus the clean tracks. One tiny, tagged silent timing track anchors zero
when real audio starts late; it is never included in the edited video. These
hidden `.mka` files stay on the selected output filesystem rather than filling
the application-data volume. Allow space for them alongside the edited videos. Scene Apply always rebuilds
from the original video and this master—not from previously blurred video or
previously scene-muted audio. Reprocessing audio also regenerates currently
approved scenes before publishing, so it does not erase the blur. Later scene
rejects/adjustments can therefore remove old blur/mute windows without losing
profanity filtering or accumulating video re-encodes. A changed source or missing
master requires reprocessing audio before another scene Apply.

Superseded masters are removed after successful processing; abandoned temporary
outputs and unreferenced masters are cleaned on the next audio run for that
title. Do not manually remove the active master if you want future scene edits
to preserve cleaned audio. Blank output settings ignore the separate-output
cache and return to the legacy source-based workflow.

The automated tests cover settings, migrations, source preservation, stream
mapping, combined-job failure, repeated processing and atomic publication with
mocked ffmpeg. Validate one audio-only and one combined audio/scene edit on real
media after deployment, including audio sync on delayed-audio sources,
re-running audio and changing scene approvals.

### Plex-compatible scene-edit interleaving

Scene-edit publication uses two **lossless stream-copy** passes. The first
interleaves the processed video and real audio tracks with early queue flushing
disabled. The second reads that AV together and restores original subtitles,
data streams, attachments, chapters, and metadata using a finite flush interval.
Sparse subtitle gaps therefore do not hold the entire movie in the zero-flush
queue. Audio titles, languages, dispositions, and attachment MIME/filename tags
are retained explicitly. Existing codec/HEVC splice and decode checks still run.

This fixes a reproduced edit whose third video packet occurred after 156 MB of
audio: Plex could encode frames but produced abnormal DASH fragmentation and
hung during playback attempts. A lossless AV repair restored the user's Plex
transcoder test to normal throughput; it does not require changing bit depth,
video codec, or GPU settings.

The AV pass temporarily needs **one additional full AV copy on the output
filesystem**. Both mux passes have a **1 GiB child-process RSS watchdog** (not a
hard allocation limit); a damaged/delayed track that exceeds the budget fails
without replacing the previous edit. Timeout/cancellation kills and reaps the
child. Hidden AV intermediates use the existing per-title orphan-sweep namespace
so interrupted runs do not leave permanent full-movie copies.

Before publication, bounded packet probes sample the beginning, middle, and end
(up to 3,000 packets per sample). Audio leading encountered video by more than
five seconds is rejected. This is a sampled layout safeguard, not a full-file
playback guarantee. Layout/resource failures do not trigger an expensive video
re-encode retry; genuine segment/decode failures retain whole-file recovery.

Rebuild the **Vulgarr server image** for this fix; worker daemons need no changes.
Existing published files are not silently rewritten: **Apply scenes** again to
regenerate an affected edit, then refresh Plex metadata and test Chrome playback.
Interrupted-job render intermediates can still be reused; completed jobs normally
remove theirs, so a fresh Apply may render again. Scan scores and approvals are
not invalidated by this mux-only change.

Opt-in synthetic media qualification (does not restart or modify worker services):

```bash
SPF_TEST_RENDER_CONTAINERS=worker-0,worker-1 python -m pytest tests/test_interleave_media.py -q
```

It covers H.264, HEVC 8/10-bit, sparse subtitles, attachments, chapters, delayed
7.1/stereo audio, and unchanged decoded video. Real affected-media Chrome/HDR
qualification remains a deployment check.

## Remote GPU workers (this fork)

See [remote_worker/README.md](remote_worker/README.md) for batch inference,
optional worker-side chunk extraction, token-authenticated media streaming,
worker status/throughput reporting, and opt-in NVENC blur-segment encoding.
All remote failures retain a local path; audio and blur keep their heavy-I/O
mutual exclusion. Remote extraction fallback reacquires that gate locally.

Build/deploy **this fork's image** to use these features; the upstream `:stable`
image in the production compose example does not include these changes. Set
`SPF_CLASSIFIER_URLS` for inference offload. Worker-side extraction also requires
`SPF_REMOTE_EXTRACT=1`, `SPF_INTERNAL_BASE_URL`, and `SPF_INTERNAL_TOKEN` on
vulgarr, plus `SPF_SOURCE_BASE_URL` on each worker. Keep worker ports private.

Remote encoding (`SPF_REMOTE_ENCODE_URL`, a worker base URL) is off by default.
Segment encoding conservatively rejects incompatible streams, especially HEVC
parameter-set mismatches. Whole-file recovery also queries the worker—even when
keyframe/split planning failed before any segment request. It tries worker NVENC,
then worker software encoding, before falling back to server CPU. This requires
`SPF_INTERNAL_BASE_URL` and `SPF_INTERNAL_TOKEN` on vulgarr and the matching
`SPF_SOURCE_BASE_URL` on the worker; remote scanning need not be enabled.
Rebuild **both** images for the pollable `/render` API. Queue messages distinguish
remote rendering, downloading, validation and server fallback. Unit coverage uses
mocked ffmpeg, **not real GPU/media qualification**: follow the worker README's
real-media validation checklist before production use.

## Features

- Poster-grid library views for movies and TV shows, synced from Sonarr/Radarr, with live search and sort
- Per-title, per-season, and per-show severity levels (Child / Teen), with whole-line or precise (word-position estimate, or Whisper forced-alignment) mute modes
- Configurable, versioned word list with severity tagging and whole-word matching
- Local, self-hosted scene detection with NudeNet labels, optional independent contextual-nudity discovery, manual review, and optional detector auto-approval
- Automatic triggers from Sonarr/Radarr import webhooks, Bazarr's subtitle-downloaded event, and/or per-title Radarr/Sonarr tags (`vulgarr-audio`/`vulgarr-video`/`vulgarr-both`, e.g. via Overseerr's Advanced Request tag picker)
- Consolidated activity queue (In Progress / Completed) spanning both pipelines, with live progress, concurrency limits, and an off-hours processing window
- Backup retention policy for original files under `/data/backups`
- Optional HTTP Basic Auth in front of the whole UI
- Runs as any PUID/PGID so files it creates match your host user

## Local dev

```bash
pip install -r requirements.txt
SPF_DATA_DIR=./data SPF_MEDIA_ROOT=./data uvicorn app.main:app --reload
```

Run tests (needs `requirements-dev.txt`, which layers `pytest` on top of the
same runtime requirements above -- also what CI installs):

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## Releasing

Every push to `main` builds and pushes `ghcr.io/dpresti/vulgarr:latest` (plus a
`sha-<short-sha>` tag) automatically -- fine for tracking `main`, but not a
real version. To cut an actual release (a clean version tag on the image, e.g.
`2.1.0`, plus a GitHub Release with auto-generated notes):

```bash
git tag -a v2.1.0 -m "v2.1.0"
git push origin v2.1.0
```

Pushing a `vX.Y.Z` tag is what triggers both of those -- a plain push to `main`
never does. A tag push also moves the `:stable` image tag to that release.
`docker-compose.yml` tracks `:stable` rather than `:latest`, so upgrading is
just `docker compose pull && up -d` and only ever picks up a real release, never
unreleased work from `main`. To pin an exact version (e.g. to roll back), set
`image:` to a specific tag like `ghcr.io/dpresti/vulgarr:2.7.0`.

## Quick start

```bash
git clone https://github.com/dpresti/vulgarr.git
cd vulgarr
cp .env.example .env
# edit .env for your setup, then:
docker compose up -d --build
```

Then open `http://<host>:<SPF_PORT>/library` (default port `8011`).

## Wiring into an existing Sonarr/Radarr/Bazarr/Plex/Jellyfin stack

1. **Compose**: [`docker-compose.yml`](docker-compose.yml) joins the same Docker network your Sonarr/Radarr/Bazarr containers are on (`media_default` by default — rename to match yours) so it's reachable from them by container name. Set these in `.env` (copy from [`.env.example`](.env.example); `.env` itself is never committed — see `.gitignore`):
   - `MEDIA_ROOT_HOST_PATH` / `SPF_MEDIA_ROOT` — the host path and matching **container-side** path for your media, which must be the exact same container-side path your Sonarr/Radarr containers already use (this app resolves file paths returned by their APIs directly, so a mismatch means it won't find files that exist). Once you've synced titles, don't change the container-side path later — it's baked into every stored path in the database.
   - `SPF_DATA_DIR` — where the SQLite DB and backups live on the host.
   - `SPF_PORT` — host port to publish the UI on (defaults to 8011).
   - `PUID` / `PGID` — the host user/group (`id -u` / `id -g`) that should own files this container creates under `SPF_DATA_DIR` and your media mounts. Defaults to 1000/1000; set these if your host user is different, otherwise you'll get permission-mismatched files.

2. **API keys**: each of Sonarr/Radarr/Bazarr has its own API key, generated by that app, under **that app's own** Settings > General > Security page (not Vulgarr's Settings) — copy it from there into `SONARR_API_KEY` / `RADARR_API_KEY` / `BAZARR_API_KEY` in Vulgarr's `.env`. Also set `SPF_WEBHOOK_TOKEN` to a random string (or leave it unset — a random one is generated automatically on first run rather than leaving webhooks unauthenticated). These `.env` values are only used to *seed* Vulgarr's database on first startup — from then on, all of them (including rotating the webhook token) are editable from Vulgarr's own **Settings > Integrations** page in the UI, no `.env` edit or restart required. Append `?token=<that value>` to the webhook URLs below, using whatever the token currently is in Vulgarr's Settings.

3. **Sonarr**: in **Sonarr's own UI**, go to Settings > Connect > add a Webhook connection. URL: `http://vulgarr:8000/webhooks/sonarr?token=<token>`. Trigger on: *On Import*, *On Upgrade*.

4. **Radarr**: same as Sonarr, in **Radarr's own UI** — Settings > Connect > add Webhook, URL: `http://vulgarr:8000/webhooks/radarr?token=<token>`.

5. **Bazarr**: in **Bazarr's own UI**, go to Settings > Subtitles > Custom Post-Processing. Enable it, and set the command to a `curl` call using Bazarr's template variables, e.g.:
   ```
   curl -s -X POST "http://vulgarr:8000/webhooks/bazarr?token=<token>&video_path={{directory}}/{{episode}}&subtitle_path={{subtitles}}&language={{subtitles_language_code2}}"
   ```
   (Bazarr's exact variable names vary by version — check Bazarr's own Settings > Subtitles > Custom Post-Processing page for the list available in your version and adjust.) Then flip "Bazarr subtitle-downloaded event" on in **Vulgarr's** Settings page if you want this trigger active.

6. **First run**: open `http://<host>:<SPF_PORT>/settings` and click "Sync from Sonarr/Radarr" to pull in the existing library (this only reads metadata/paths via the Sonarr/Radarr APIs — it does not process anything). Then check `/library/movies` or `/library/shows` to see the synced titles, and go to `/wordlist` to build out your word list before processing anything for real. Scene detection needs no separate setup — it's manual-start per title/season/show from the library, with its own tuning under Settings > Video Processing if you want it.

7. **Plex/Jellyfin**: nothing to configure — once a title has been audio-processed, refresh/rescan that item's metadata (or wait for the next library scan) and the "Clean" audio track will appear in the audio track picker like any other track. A "Vulgarr Edit" from the video pipeline shows up as a separate playable file/version, same as any other extra cut Plex/Jellyfin already knows how to list.

## Settings reference

The Settings page is grouped into sections (Integrations, Audio Processing,
Video Processing, Backups, Authentication), with a "Show advanced settings"
toggle hiding the finer classifier/encode-tuning knobs by default:

- **Integrations** — Sonarr/Radarr/Bazarr URLs+API keys, the webhook token, default subtitle language, and the clean track's title/language tag. Editable here at any time; `.env` only supplies the first-run defaults.
- **Audio Processing** — max concurrent ffmpeg jobs, off-hours processing window, automation triggers (independently enable/disable Sonarr/Radarr import and Bazarr's subtitle-downloaded event, and which one takes priority for de-duplication), and the default mute precision for newly-synced titles.
- **Video Processing** — local-only contextual discovery (endpoint, optional key, model, sampling and timeout; off by default), separate legacy candidate verification, detector auto-approval, classifier tuning, blur quality/speed/intensity with a live preview, and blur/mute padding. See [local contextual discovery](docs/LOCAL_CONTEXTUAL_NUDITY.md) for setup and limitations. Optional [protection-first automation](docs/PROTECTION_FIRST_AUTOMATION.md) adds conservative automatic decisions, denser/shot-based sampling, local rechecks, automatic rendering and staged-output qualification into the configured clean folders, without Plex library changes.
- **Backups** — retention (in days) for files under `/data/backups`; defaults to 7, 0 keeps every backup forever.
- **Authentication** — optional HTTP Basic Auth in front of the entire UI (**off by default**). Turn this on if the app is reachable by anyone besides you — e.g. exposed outside your LAN — since nothing here is protected otherwise, including the Sonarr/Radarr/Bazarr API keys visible on this same Settings page. Sonarr/Radarr/Bazarr's webhook calls are always exempt (they carry their own `?token=`, not a login). Note: Basic Auth alone doesn't protect against CSRF (browsers resend cached credentials on cross-site requests) — fine for a trusted LAN, but if you're exposing this beyond that, put it behind a reverse proxy/VPN rather than relying on Basic Auth as your only barrier.

## Notes / known trade-offs

- Backups accumulate under `/data/backups` for 7 days by default (Settings > Backups); set it to 0 to keep every backup forever instead.
- `backup_root` (`/data/backups`) should ideally be on the same filesystem as your media mount; otherwise the "move original aside" step becomes a slower copy+delete instead of an instant rename.
- Subtitle matching mutes the whole subtitle cue's time range (padded slightly) by default — a plain `.srt` has no word-level timestamps of its own. Settings > Audio Processing has two opt-in, narrower alternatives: `estimate` (a proportional guess based on the word's position in the cue's text) and `whisper` (real word-level forced alignment against the actual audio, not a guess).
- Scene detection is not a certainty: sampling can miss short shots, and vision models can misread clothing or bare skin. Contextual candidates require review unless protection-first automation is explicitly enabled. That mode automatically filters unresolved ambiguity and holds failed/incomplete results rather than publishing a new copy; it can over-blur and is not a guarantee of recall. Whole-title contextual discovery can be slow on CPU. Neither a contextual flag nor a model's self-reported confidence is a guarantee of accuracy.
- There's no per-user accounts/roles — Settings > Authentication is a single shared username/password (or nothing) in front of the whole app, not a multi-user system.

## License

[GPL-3.0](LICENSE)
