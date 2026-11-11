"""Second, small job dispatcher for scene-detection work (scan now, blur in a
later phase). Deliberately not merged into app.queue.worker's JobQueue -- that
class already drives all of the proven audio-pipeline's behavior/tests, and
generalizing it to be job-kind-polymorphic is a real refactor of working code
for a mostly-aesthetic benefit. A little duplication (this file) is accepted
instead; see the scene-detection plan for the reasoning."""

import asyncio
import datetime
import logging
import shutil
import time
from pathlib import Path
from dataclasses import dataclass

from sqlalchemy import select, update, delete

from app.config import settings as app_settings
from app.db.models import JobState, SceneJob, SceneJobKind, Title, DetectedScene, ProcessingJob
from app.db.session import get_session, get_setting
from app.mux.tempfiles import sweep_interrupted_temp_files
from app.queue.heavy_pipeline_lock import HEAVY_PIPELINE_LOCK
from contextlib import nullcontext
from app.vision.remote_source import remote_extract_enabled
from app.queue.worker import is_within_off_hours_window
from app.scenes.pipeline import apply_scene_blur, reverify_scenes_with_claude, scan_for_scenes

logger = logging.getLogger(__name__)

_PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 1.5

# A scan's per-frame ETA is far more stable than the audio pipeline's ffmpeg-progress
# ETA (uniform-cost work, not variable-length cues), so no warm-up-fraction gating is
# needed here -- shown as soon as there's more than one frame's worth of data.


def _parse_hhmm(value: str) -> datetime.time:
    hour, minute = value.split(":")
    return datetime.time(hour=int(hour), minute=int(minute))


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m remaining"
    if minutes:
        return f"{minutes}m {seconds}s remaining"
    return f"{seconds}s remaining"


@dataclass(frozen=True)
class SceneResetResult:
    """Committed outcome of an explicit destructive reset, not a normal rescan."""
    restarted: int = 0
    cleared: int = 0
    busy: int = 0
    missing: int = 0

    def message(self) -> str:
        return (
            f"Cleared {self.cleared} scene(s); restarted scanning from zero for {self.restarted} title(s). "
            + (f"Skipped {self.busy} busy title(s) with queued/running audio, Apply or verification work. " if self.busy else "")
            + (f"{self.missing} title(s) no longer available. " if self.missing else "")
            + "Published edits are retained until a validated regeneration succeeds."
        )


class SceneJobQueue:
    def __init__(self) -> None:
        self._pending: asyncio.Queue[int] = asyncio.Queue()
        self._running: dict[int, asyncio.Task] = {}
        self._dispatcher_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        # Re-queue anything left mid-flight from a prior process crash/restart.
        async with get_session() as session:
            result = await session.execute(
                select(SceneJob).where(SceneJob.state.in_([JobState.queued, JobState.processing]))
            )
            active_scan_title_ids: set[int] = set()
            interrupted_videos: list[Path] = []
            for job in result.scalars().all():
                was_processing = job.state == JobState.processing
                job.state = JobState.queued
                job.started_at = None
                title = await session.get(Title, job.title_id)
                if title is not None:
                    title.scene_scan_status = "queued"
                    if was_processing and job.kind == SceneJobKind.blur:
                        interrupted_videos.append(Path(title.video_path))
                if job.kind == SceneJobKind.scan:
                    active_scan_title_ids.add(job.title_id)
                    # Upgrade interrupted pre-generation jobs in place.
                    if title is not None and title.scene_scan_generation is None:
                        title.scene_scan_generation = job.id

            # Defensive reconciliation: a Title stuck showing "queued"/"scanning"
            # with no actual SceneJob behind it (the requeue loop above only
            # re-queues real orphaned *jobs*; this catches an orphaned *status*
            # with no job at all, e.g. from a job being deleted/edited outside
            # the normal cancel_job()/_run_job() paths, which are the only two
            # places that otherwise keep this field in sync). Not expected in
            # normal operation, but cheap to check and prevents a permanently
            # stuck-looking scan indicator either way.
            result = await session.execute(select(Title).where(Title.scene_scan_status.in_(["queued", "scanning"])))
            for title in result.scalars().all():
                if title.id not in active_scan_title_ids:
                    title.scene_scan_status = "not_scanned"

            await session.commit()

            result = await session.execute(select(SceneJob.id).where(SceneJob.state == JobState.queued))
            for (job_id,) in result.all():
                await self._pending.put(job_id)

        # Same as JobQueue.start(): an interrupted blur's temp output is orphaned.
        await sweep_interrupted_temp_files(interrupted_videos)

        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        for task in list(self._running.values()):
            task.cancel()

    async def cancel_job(self, job_id: int) -> tuple[bool, str | None]:
        """Mirrors JobQueue.cancel_job -- same >=97% "too far along" refusal, more
        important here than for a scan (harmless to abandon mid-way) once blur jobs
        (a real file-swap tail step) start using this same queue/dispatcher."""
        task = self._running.get(job_id)
        if task is not None:
            async with get_session() as session:
                job = await session.get(SceneJob, job_id)
                # A task is registered here as soon as it's created, before its
                # own body has necessarily run far enough to flip state to
                # "processing" -- treat that brief "queued" window the same as
                # "processing" rather than rejecting the cancel (see
                # JobQueue.cancel_job for the same fix/reasoning).
                if job is None or job.state not in (JobState.queued, JobState.processing):
                    return False, "Job is no longer running"
                if job.state == JobState.processing and (job.progress_percent or 0) >= 97:
                    return False, "Too far along to cancel -- finishing up"
            task.cancel()
            return True, None

        async with get_session() as session:
            job = await session.get(SceneJob, job_id)
            if job is None or job.state != JobState.queued:
                return False, "Job is not queued or running"
            job.state = JobState.cancelled
            job.error = "Cancelled by user"
            job.finished_at = datetime.datetime.utcnow()
            title = await session.get(Title, job.title_id)
            if title is not None and job.kind == SceneJobKind.scan:
                title.scene_scan_status = "not_scanned"
            await session.commit()
        return True, None

    async def clear_and_restart_scans(self, title_ids: list[int]) -> SceneResetResult:
        """Explicitly clear current windows now and start new inference generations.

        Unlike ordinary atomic rescans, this destructive action intentionally
        leaves no current scene results even if subsequent inference fails.
        Original media, published edits, clean masters and profanity decisions
        are untouched. Existing scans are superseded transactionally; active
        audio/render/verification jobs are skipped without deleting anything.
        New job IDs guarantee fresh checkpoints rather than retry/resume.
        """
        new_jobs: list[int] = []
        superseded: list[int] = []
        cleared = busy = missing = 0
        async with get_session() as session:
            for title_id in sorted(set(title_ids)):
                scene_busy = select(SceneJob.id).where(
                    SceneJob.title_id == title_id, SceneJob.kind != SceneJobKind.scan,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                ).exists()
                audio_busy = select(ProcessingJob.id).where(
                    ProcessingJob.title_id == title_id,
                    ProcessingJob.state.in_([JobState.queued, JobState.processing]),
                ).exists()
                # Acquire the database writer reservation before checking and
                # replacing the old generation. Concurrent Apply claims use the
                # same Title write and cannot race this check/delete operation.
                claim = await session.execute(update(Title).where(
                    Title.id == title_id, ~scene_busy, ~audio_busy,
                ).values(scene_scan_status="queued", scene_analysis=None))
                if not claim.rowcount:
                    if await session.get(Title, title_id) is None:
                        missing += 1
                    else:
                        busy += 1
                    continue
                active = (await session.execute(select(SceneJob.id).where(
                    SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.scan,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                ))).scalars().all()
                if active:
                    await session.execute(update(SceneJob).where(SceneJob.id.in_(active)).values(
                        state=JobState.cancelled, error="Superseded by Clear scenes & restart scan",
                        finished_at=datetime.datetime.utcnow(),
                    ))
                    superseded.extend(active)
                deleted = await session.execute(delete(DetectedScene).where(DetectedScene.title_id == title_id))
                cleared += deleted.rowcount
                job = SceneJob(title_id=title_id, kind=SceneJobKind.scan, state=JobState.queued,
                    keep_manual=False, progress_message="Previous scenes cleared; fresh scan queued", progress_percent=0)
                session.add(job)
                await session.flush()
                await session.execute(update(Title).where(Title.id == title_id)
                                      .values(scene_scan_generation=job.id))
                new_jobs.append(job.id)
            await session.commit()
        # Cancel local tasks only after the durable fence is installed. Other
        # processes' old scans cannot activate against the replacement generation.
        for job_id in superseded:
            task = self._running.get(job_id)
            if task is not None:
                task.cancel()
        for job_id in new_jobs:
            await self._pending.put(job_id)
        return SceneResetResult(len(new_jobs), cleared, busy, missing)

    async def enqueue_scan(self, title_id: int, *, keep_manual: bool = False) -> int:
        """Claim a fresh generation using a database write, not a check-then-insert.

        SQLite serializes the conditional UPDATE across processes. The title
        claim and job creation commit together, including for bulk callers.
        Zero means the title is missing or already has an active scene job.
        """
        async with get_session() as session:
            active = select(SceneJob.id).where(
                SceneJob.title_id == title_id,
                SceneJob.state.in_([JobState.queued, JobState.processing]),
            ).exists()
            claimed = await session.execute(
                update(Title).where(Title.id == title_id, ~active)
                .values(scene_scan_status="queued")
            )
            if not claimed.rowcount:
                return 0
            job = SceneJob(title_id=title_id, kind=SceneJobKind.scan,
                           state=JobState.queued, keep_manual=keep_manual)
            session.add(job)
            await session.flush()
            await session.execute(update(Title).where(Title.id == title_id)
                                  .values(scene_scan_generation=job.id))
            job_id = job.id
            await session.commit()
        await self._pending.put(job_id)
        return job_id

    async def enqueue_scan_if_not_already_active(self, title_id: int) -> bool:
        """Mirrors app.library.enqueue_if_not_already_active -- used by both the
        single-title scan button and the bulk season/show scan actions, so a
        double-click or an overlapping bulk trigger doesn't queue a second scan for
        a title that's already queued/processing. Returns whether it actually
        enqueued (vs. skipped as a duplicate)."""
        return bool(await self.enqueue_scan(title_id))

    async def enqueue_scan_for_titles_if_not_already_active(self, title_ids: list[int]) -> int:
        """Claim each distinct title atomically through the shared enqueue path."""
        return sum([bool(await self.enqueue_scan(tid)) for tid in dict.fromkeys(title_ids)])

    async def enqueue_blur(self, title_id: int) -> int:
        """Atomically exclude scans and other scene work while applying a set."""
        return await self._enqueue_existing_set_job(title_id, SceneJobKind.blur)

    async def _enqueue_existing_set_job(self, title_id: int, kind: SceneJobKind) -> int:
        async with get_session() as session:
            active = select(SceneJob.id).where(SceneJob.title_id == title_id,
                SceneJob.state.in_([JobState.queued, JobState.processing])).exists()
            claimed = await session.execute(update(Title).where(Title.id == title_id, ~active)
                .values(scene_scan_status=Title.scene_scan_status))
            if not claimed.rowcount:
                return 0
            job = SceneJob(title_id=title_id, kind=kind, state=JobState.queued)
            session.add(job)
            await session.flush()
            job_id = job.id
            await session.commit()
        await self._pending.put(job_id)
        return job_id

    async def enqueue_blur_if_not_already_active(self, title_id: int) -> bool:
        """Same dedup reasoning as enqueue_scan_if_not_already_active -- a
        double-click on "Apply" shouldn't queue a second multi-hour re-encode for
        the same title."""
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob).where(
                    SceneJob.title_id == title_id,
                    SceneJob.kind == SceneJobKind.blur,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            if existing.scalar_one_or_none() is not None:
                return False
        return bool(await self.enqueue_blur(title_id))

    async def enqueue_claude_verify(self, title_id: int) -> int:
        return await self._enqueue_existing_set_job(title_id, SceneJobKind.claude_verify)

    async def enqueue_claude_verify_if_not_already_active(self, title_id: int) -> bool:
        """Same dedup reasoning as enqueue_scan_if_not_already_active -- a double-click
        on "Reprocess with Claude Vision" shouldn't queue a second pass over the same
        title's candidates while one is already running."""
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob).where(
                    SceneJob.title_id == title_id,
                    SceneJob.kind == SceneJobKind.claude_verify,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            if existing.scalar_one_or_none() is not None:
                return False
        return bool(await self.enqueue_claude_verify(title_id))

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            try:
                async with get_session() as session:
                    cap = int(await get_setting(session, "scene_scan_concurrency_cap"))
                    off_hours_enabled = bool(await get_setting(session, "off_hours_enabled"))
                    off_start = _parse_hhmm(await get_setting(session, "off_hours_start"))
                    off_end = _parse_hhmm(await get_setting(session, "off_hours_end"))

                allowed_to_run = True
                if off_hours_enabled:
                    now = datetime.datetime.now().time()
                    allowed_to_run = is_within_off_hours_window(now, off_start, off_end)

                if allowed_to_run:
                    while len(self._running) < cap and not self._pending.empty():
                        job_id = self._pending.get_nowait()
                        task = asyncio.create_task(self._run_job(job_id))
                        self._running[job_id] = task
                        task.add_done_callback(lambda t, jid=job_id: self._running.pop(jid, None))

                await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scene dispatcher loop error")
                await asyncio.sleep(5)

    async def _run_job(self, job_id: int) -> None:
        async with get_session() as session:
            # Duplicate queue deliveries must not start two inference/render
            # pipelines. Restart reconciliation intentionally restores queued.
            claimed = await session.execute(update(SceneJob).where(
                SceneJob.id == job_id, SceneJob.state == JobState.queued,
            ).values(state=JobState.processing, started_at=datetime.datetime.utcnow()))
            if not claimed.rowcount:
                return
            job = await session.get(SceneJob, job_id)
            title_id = job.title_id
            title = await session.get(Title, title_id)
            if title is None:
                job.state = JobState.failed
                job.error = "Title no longer exists"
                job.finished_at = datetime.datetime.utcnow()
                await session.commit()
                return

            is_scan = job.kind == SceneJobKind.scan
            is_claude_verify = job.kind == SceneJobKind.claude_verify
            job.state = JobState.processing
            job.started_at = datetime.datetime.utcnow()
            job.progress_percent = 0.0
            if is_scan:
                job.progress_message = "Scanning for scenes"
                title.scene_scan_status = "scanning"
            elif is_claude_verify:
                job.progress_message = "Verifying scenes with Claude Vision"
            else:
                job.progress_message = "Blurring video"
            await session.commit()

            job_start_monotonic = time.monotonic()
            last_update_monotonic = 0.0
            scan_phase = "Scanning for scenes"
            blur_phase = "Blurring video (ffmpeg)"
            scan_counters = ""

            async def on_scan_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                fraction = done / total
                job.progress_percent = round(100 * fraction, 1)
                message = scan_phase
                if fraction > 0 and not is_final:
                    elapsed = now - job_start_monotonic
                    eta_seconds = elapsed * (1 - fraction) / fraction
                    message += f", {_format_eta(eta_seconds)}"
                job.progress_message = message
                await session.commit()

            # Unlike app.queue.worker's mute pipeline (one whole-file ffmpeg pass,
            # where a 97% floor reserves the last few points for the post-encode
            # verify/publish stage below), app.mux.scene_blur's split/re-encode/
            # concat pipeline calls on_stage throughout the whole job (once per
            # scene segment, not just at the tail), so on_stage below no longer
            # forces a progress floor -- fraction naturally reaches ~97% once all
            # scene segments are done re-encoding, before verify/publish ever run.
            # (When any approved scene has mute_audio set, apply_scene_blur scales
            # this fraction down to leave room for the audio-mute phase that
            # follows -- see its docstring -- so this doesn't hit 97% until that
            # phase, which on_stage alone doesn't move progress_percent for, has
            # also actually finished.)
            async def on_blur_progress(fraction: float) -> None:
                nonlocal last_update_monotonic
                now = time.monotonic()
                is_final = fraction >= 0.999
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                job.progress_percent = round(fraction * 97.0, 1)
                message = f"{blur_phase} -- {round(fraction * 100):.0f}%"
                if fraction > 0.02 and not is_final:
                    elapsed = now - job_start_monotonic
                    eta_seconds = elapsed * (1 - fraction) / fraction
                    message += f", {_format_eta(eta_seconds)}"
                job.progress_message = message
                await session.commit()

            async def on_blur_stage(message: str) -> None:
                nonlocal blur_phase
                blur_phase = message
                job.progress_message = message
                await session.commit()

            async def on_scan_stage(message: str) -> None:
                nonlocal scan_phase, scan_counters
                if ' · remote ' in message:
                    # Retain the latest counters through dense rescans/VLM
                    # stages and in the terminal summary after polling swaps.
                    telemetry = message.split(' · remote ', 1)[1]
                    scan_counters = ' · remote ' + ' · '.join(telemetry.split(' · ')[:2])
                scan_phase = message
                if message.startswith(("Extracting chunk", "Classifying ")):
                    return  # The paired progress callback throttles database writes.
                # No progress-percent bump like on_blur_stage's 97% floor -- the
                # frame-count progress above already reaches 100% once the main
                # scan finishes, so the per-candidate verification pass that
                # follows is a (typically short) tail step, not something that
                # needs its own slice of the bar.
                job.progress_message = message + scan_counters
                await session.commit()

            async def on_claude_verify_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                job.progress_percent = round(100 * done / total, 1)
                job.progress_message = f"Verifying with Claude Vision -- scene {done}/{total}"
                await session.commit()

            try:
                if is_scan:
                    # Local fallback reacquires the gate around extraction;
                    # audio and blur still retain whole-pipeline exclusion.
                    async with (nullcontext() if remote_extract_enabled() else HEAVY_PIPELINE_LOCK):
                        scan_outcome = await scan_for_scenes(
                            session, title, job.id, on_progress=on_scan_progress, on_stage=on_scan_stage
                        )
                    job.progress_message = (
                        f"{scan_outcome.candidate_count} candidates · "
                        f"{scan_outcome.auto_approved_count} auto-approved · "
                        f"{scan_outcome.need_review_count} need review"
                    ) + scan_counters
                    title.scene_scan_status = "scanned" if scan_outcome.candidate_count else "no_scenes_found"

                    # Chain straight into a blur/Apply job when auto-process is on
                    # and this scan actually found something -- scan_for_scenes
                    # already auto-approved the candidates above (same setting
                    # gates both halves), this just closes the loop so the user
                    # doesn't have to separately click Apply. Committed before
                    # enqueuing so the blur job's own session sees the scenes as
                    # approved. A concurrent manual scan-scenes/apply-scenes click
                    # can't double-enqueue here -- enqueue_blur_if_not_already_active
                    # checks for an existing queued/processing blur job first.
                    if (scan_outcome.candidate_count or title.vulgarr_edit_path) and bool(await get_setting(session, "scene_auto_process")):
                        await session.commit()
                        await self.enqueue_blur_if_not_already_active(title.id)
                elif is_claude_verify:
                    async with HEAVY_PIPELINE_LOCK:
                        reverify_outcome = await reverify_scenes_with_claude(
                            session, title, on_progress=on_claude_verify_progress
                        )
                    job.progress_message = (
                        f"Done -- checked {reverify_outcome.checked_count} scene(s), "
                        f"{reverify_outcome.approved_count} confirmed, "
                        f"{reverify_outcome.rejected_count} rejected"
                        + (f", {reverify_outcome.failed_count} failed" if reverify_outcome.failed_count else "")
                    )
                else:
                    async with HEAVY_PIPELINE_LOCK:
                        # An audio job can install a new clean master while
                        # this job waits for the gate. Its pre-wait identity-map
                        # snapshot must not reference the now-retired cache.
                        await session.refresh(title, attribute_names=[
                            'clean_audio_path', 'clean_audio_source_fingerprint',
                            'source_clean_track_audio_indices',
                        ])
                        blur_outcome = await apply_scene_blur(
                            session, title, job.id, on_progress=on_blur_progress, on_stage=on_blur_stage
                        )
                    job.progress_message = f"Done -- blurred {blur_outcome.scene_count} scene(s)"

                job.state = JobState.done
                job.progress_percent = 100.0
                job.finished_at = datetime.datetime.utcnow()

            except asyncio.CancelledError:
                await session.rollback()
                job = await session.get(SceneJob, job_id, populate_existing=True)
                title = await session.get(Title, title_id, populate_existing=True)
                if job.state == JobState.done:
                    raise
                job.state = JobState.cancelled
                job.error = "Cancelled by user"
                job.finished_at = datetime.datetime.utcnow()
                if is_scan and title is not None and title.scene_scan_generation == job_id:
                    title.scene_scan_status = "not_scanned"
                await session.commit()
                raise  # required so the Task itself is properly marked cancelled

            except Exception as exc:  # noqa: BLE001 -- surface all failures to the UI
                await session.rollback()
                job = await session.get(SceneJob, job_id, populate_existing=True)
                title = await session.get(Title, title_id, populate_existing=True)
                if job.state == JobState.cancelled:
                    # A reset from another process can fence activation without
                    # cancelling this process's Task. Keep the intentional
                    # cancellation result rather than relabeling it as a failure.
                    return
                logger.exception("Scene %s failed for title %s", job.kind.value, title_id)
                job.state = JobState.failed
                job.error = str(exc)
                job.finished_at = datetime.datetime.utcnow()
                if is_scan and title is not None and title.scene_scan_generation == job_id:
                    title.scene_scan_status = "scan_failed"
                if job.kind == SceneJobKind.blur:
                    # A failed blur job is terminal -- apply_scene_blur's docstring
                    # only resumes a work_dir when the *same* job_id gets
                    # redispatched after an interrupted (CancelledError) run; a
                    # retry after a genuine failure is always a fresh Apply click
                    # with a new job_id, so this one's work_dir is never coming
                    # back for. Left behind otherwise, this is a multi-GB ffmpeg
                    # scratch dir that sits on disk forever.
                    work_dir = app_settings.data_dir / "blur_work" / str(job.id)
                    await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)

            await session.commit()


scene_job_queue = SceneJobQueue()
