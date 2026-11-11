# Clear scenes & restart scan

An explicit **Clear scenes & restart scan** action is available on:

- Movie/episode table rows and title-detail pages.
- Show-detail pages, affecting all available episodes of that show.
- Season rows/detail pages, affecting only that season.
- Standalone scene-review pages.

It appears separately from the Audio / Video / Both buttons. It does not require subtitles or reset profanity/audio decisions.

## Semantics

This is deliberately different from a normal scan/reprocess:

- **Normal scan:** retain previous scene results until successful replacement.
- **Clear scenes & restart scan:** immediately delete all current scene windows and review decisions—pending, approved, rejected and manual—then queue fresh inference from zero.

The confirmation explicitly explains the destructive behavior. If the replacement scan fails or is cancelled, the intentionally cleared scene rows are not restored. Original media, cached clean audio, profanity decisions, and the last published edited file remain untouched by the reset. An edited file changes only after validated regeneration succeeds through the existing rendering pipeline (automatic Apply still follows the title's configured auto-process behavior).

Every restarted title gets a new scan job/generation ID and therefore a new checkpoint directory. It never resumes the old job's partial scores. Existing queued/running scans are cancelled/superseded transactionally; their late completion cannot publish scenes into the replacement generation. Local running tasks are cancelled only after that database fence commits.

Titles with queued/running audio, Apply or external-verification jobs are skipped without clearing any decisions. Bulk requests report how many scenes were cleared, how many titles restarted, and how many busy titles were skipped. Other movies/shows/seasons are never affected. Show/season context is reloaded after resetting so the refreshed UI is not built from pre-reset data.

Confirmation is also required by the server (`confirm_reset=true`), not just a browser dialog. The action does not deploy workers, change global preferences or restart services.

## Regression coverage

`tests/test_scene_reset.py` covers immediate deletion of all review states/manual windows, media and audio-decision preservation, fresh job identities, queued/running scan supersession, stale activation refusal, concurrent resets, busy-job protection, movie/show/season scoping, partial bulk skips, missing titles, row/detail htmx targets and mandatory confirmation.

```bash
python -m pytest tests/ -q
# 453 passed, 12 skipped

SPF_TEST_RENDER_CONTAINERS=vulgarr-worker-worker-0-1,vulgarr-worker-worker-1-1 python -m pytest tests/ -q
# 477 passed
```

Tests use isolated SQLite databases and synthetic/diagnostic media only. No existing user media was reset or regenerated during implementation.
