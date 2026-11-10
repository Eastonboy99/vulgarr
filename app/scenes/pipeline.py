"""Ties video frame classification -> clustering -> DetectedScene persistence
together for one title's scene-detection scan. Mirrors app.processing's role for
the subtitle/mute pipeline, but is a wholly independent feature -- see the
scene-detection plan for why this doesn't share Title.status/ProcessingJob."""

import datetime
import hashlib
import json
import logging
import shutil
import secrets
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.mute import MuteInterval
from app.common.intervals import merge_intervals
from app.config import settings as app_settings
from app.db.models import DetectedScene, Title, SceneJob, JobState
from app.db.session import get_setting
from app.domain import SceneReviewStatus
from app.mux.remux import ProgressCallback, StageCallback, probe
from app.mux.scene_blur import apply_blur
from app.scenes.content_policy import Action, VISUAL_CATEGORIES, NUDENET_CATEGORIES, RENDER_SETTING_KEYS, scene_decision, policy_fingerprint, scene_set_fingerprint, validate_policy, nudenet_evidence
from app.vision.nudenet_labels import NUDENET_LABELS, category_id
from app.vision.content_classifier import scan_content_windows, classify_window
from app.vision.claude_verify import verify_candidate
from app.vision.classifier import scan_video_frames, scan_window_frames, detection_labels
from app.vision.scene_cluster import (
    boundary_touches_window_edge,
    cluster_scenes,
    refine_scene_boundary,
    verified_fraction,
)

logger = logging.getLogger(__name__)

ScanProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class ScanOutcome:
    candidate_count: int
    auto_approved_count: int = 0
    need_review_count: int = 0


@dataclass(frozen=True)
class BlurOutcome:
    output_path: Path
    scene_count: int


@dataclass(frozen=True)
class ReverifyOutcome:
    checked_count: int
    approved_count: int
    rejected_count: int
    failed_count: int


def _scan_fingerprint(video_path: Path, frame_interval_seconds: float, duration: float) -> str:
    """Guards a scan's resume checkpoint the same way _blur_job_fingerprint
    guards the blur pipeline's work_dir: frame timestamps in the checkpoint
    are computed as index*frame_interval_seconds, so if that setting (or the
    source file itself) changed between an interrupted attempt and its
    resume, blindly trusting the old checkpoint's timestamps would silently
    corrupt them rather than just costing a little redundant work -- treated
    as fully stale and wiped instead."""
    stat = video_path.stat() if video_path.exists() else None
    try:
        detector_version = version("nudenet")
    except PackageNotFoundError:
        detector_version = "3.4.2"  # Pinned identically in app/worker requirements.
    payload = repr(("nudenet-scores-v2", detector_version, str(video_path),
                    (stat.st_size, stat.st_mtime_ns) if stat else None,
                    frame_interval_seconds, round(duration, 1)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def _refine_candidate_boundary(
    video_path: Path,
    candidate_start: float,
    candidate_end: float,
    verify_pad: float,
    verify_max_pad: float,
    verify_interval: float,
    confidence_threshold: float,
    frame_concurrency: int,
) -> tuple[float, float, float | None, list]:
    """Dense re-scan of a candidate's padded window (app.vision.classifier.
    scan_window_frames), doubling the pad outward -- up to verify_max_pad -- and
    re-scanning whenever the refined boundary (app.vision.scene_cluster.
    refine_scene_boundary) lands right at the edge of what was actually searched.

    Exists because a boundary at the search window's own wall is indistinguishable
    from "the scene genuinely ends here" only by coincidence (see
    boundary_touches_window_edge's docstring) -- without this, a scene whose real
    extent runs past the fixed verify_pad is silently clipped there every time,
    which is exactly the "blur starts after nudity is already on screen / stops
    before the scene ends" failure real usage reported (confirmed independently by
    this app's own benchmark harness: mean boundary error of ~31s across a real
    25-episode GoT sample, far more than a single fixed 5s pad could ever explain).

    Returns (start, end, verified_fraction, last successful pass's window_scores).
    verified_fraction and window_scores come from the *widest* pass that actually
    ran, not necessarily the one whose boundary was kept -- there isn't a case
    where a later pass finds a narrower real boundary than an earlier one already
    did, since each pass only ever searches a superset of the previous pass's
    window.

    Falls back to the original coarse candidate boundary, unchanged, if the very
    first pass fails outright or finds nothing above threshold -- same
    fail-gracefully contract callers had before this function existed. A failure
    on a *later* (expanded) pass instead keeps whatever the last successful pass
    already found, rather than discarding it -- an expansion attempt failing
    shouldn't cost the candidate a perfectly good narrower result it already had.
    """
    start, end, frac = candidate_start, candidate_end, None
    window_scores: list = []
    pad = verify_pad
    while True:
        try:
            new_window_scores = await scan_window_frames(
                video_path,
                app_settings.ffmpeg_bin,
                window_start=candidate_start,
                window_end=candidate_end,
                pad_seconds=pad,
                frame_interval_seconds=verify_interval,
                concurrency=frame_concurrency,
            )
        except Exception:
            logger.warning(
                "Scene verification pass failed for %s at %.1f-%.1fs (pad=%.1fs)",
                video_path, candidate_start, candidate_end, pad, exc_info=True,
            )
            break
        window_scores = new_window_scores
        frac = verified_fraction(window_scores, confidence_threshold)
        boundary = refine_scene_boundary(window_scores, confidence_threshold)
        if boundary is None:
            break
        start, end = boundary
        window_start = max(0.0, candidate_start - pad)
        window_end = candidate_end + pad
        touches_start, touches_end = boundary_touches_window_edge(
            boundary, window_start, window_end, tolerance=verify_interval * 1.5
        )
        if (not touches_start and not touches_end) or pad >= verify_max_pad:
            break
        pad = min(pad * 2, verify_max_pad)
    return start, end, frac, window_scores


async def scan_for_scenes(session: AsyncSession, title: Title, job_id: int,
                          on_progress: ScanProgressCallback | None = None,
                          on_stage: StageCallback | None = None) -> ScanOutcome:
    """Apply the title scan's exact NudeNet proposal classes without global state."""
    policy = validate_policy(await get_setting(session, "scene_content_policy"))
    selected = [label for label in NUDENET_LABELS if policy["actions"][category_id(label)] != "allow"]
    with detection_labels(selected):
        return await _scan_for_scenes(session, title, job_id, on_progress, on_stage)


async def _scan_for_scenes(
    session: AsyncSession,
    title: Title,
    job_id: int,
    on_progress: ScanProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> ScanOutcome:
    """Scan title's video for candidate explicit-content scenes.

    Builds candidates entirely in memory, then atomically replaces the prior
    scene set, including approved and manual windows by default. A job's
    explicit keep_manual flag preserves only manual windows. Progress commits
    cannot publish partial results. The durable requested generation fences
    late, cancelled, or superseded scans before any row is removed.

    job_id keys the resumable checkpoint file (app_settings.data_dir /
    "scan_work" / str(job_id) / "scores.jsonl") the main frame scan uses --
    see app.vision.classifier._scan_frames. Same reasoning as
    apply_scene_blur's work_dir: scene_worker's startup reconciliation
    redispatches an interrupted job under this same id, so a resumed attempt
    naturally finds and continues from its own prior checkpoint; a genuinely
    new scan (new job_id) always starts clean.

    Each candidate the main scan finds then gets a second, much denser re-scan
    of just its own padded window (scan_window_frames) -- cheap, since it's a
    handful of short clips rather than the whole movie, and produces a
    verified_fraction (see scene_cluster.verified_fraction) that's a more
    robust per-scene signal than the main scan's single peak_confidence value.
    Powers the review list's one-click "Approve high-confidence" bulk action.
    """
    auto_process = bool(await get_setting(session, "scene_auto_process"))
    confidence_threshold = float(await get_setting(session, "scene_confidence_threshold"))
    frame_interval = float(await get_setting(session, "scene_frame_interval_seconds"))
    min_duration = float(await get_setting(session, "scene_min_duration_seconds"))
    merge_gap = float(await get_setting(session, "scene_merge_gap_seconds"))
    frame_concurrency = int(await get_setting(session, "scene_frame_classify_concurrency"))
    verify_pad = float(await get_setting(session, "scene_verify_pad_seconds"))
    verify_interval = float(await get_setting(session, "scene_verify_frame_interval_seconds"))
    verify_max_pad = float(await get_setting(session, "scene_verify_max_pad_seconds"))
    high_confidence_override = float(await get_setting(session, "scene_high_confidence_single_frame_threshold"))

    video_path = Path(title.video_path)
    src_probe = await probe(app_settings.ffprobe_bin, video_path)
    duration = float(src_probe["format"].get("duration", 0))

    scan_work_dir = app_settings.data_dir / "scan_work" / str(job_id)
    scan_work_dir.mkdir(parents=True, exist_ok=True)
    resume_scores_path = scan_work_dir / "scores.jsonl"
    fingerprint_path = scan_work_dir / "fingerprint.txt"
    policy = validate_policy(await get_setting(session, "scene_content_policy"))
    selected = sorted(k for k in NUDENET_CATEGORIES if policy["actions"][k] != "allow")
    fingerprint = _scan_fingerprint(video_path, frame_interval, duration)
    fingerprint += hashlib.sha256(repr(selected).encode()).hexdigest()[:16]
    if not fingerprint_path.exists() or fingerprint_path.read_text().strip() != fingerprint:
        resume_scores_path.unlink(missing_ok=True)
        fingerprint_path.write_text(fingerprint)

    if bool(await get_setting(session, "scene_semantic_enabled")):
        return await _scan_semantic(session, title, job_id, scan_work_dir,
                                    fingerprint, duration, on_progress, on_stage)

    scores = await scan_video_frames(
        video_path,
        app_settings.ffmpeg_bin,
        duration_seconds=duration,
        frame_interval_seconds=frame_interval,
        on_progress=on_progress,
        concurrency=frame_concurrency,
        resume_scores_path=resume_scores_path,
        on_status=on_stage,
    )

    candidates = cluster_scenes(
        scores,
        confidence_threshold=confidence_threshold,
        frame_interval_seconds=frame_interval,
        merge_gap_seconds=merge_gap,
        high_confidence_override=high_confidence_override,
    )

    # Each candidate's dense re-scan drives two things from the same data: the
    # verified_fraction confidence signal (unchanged), and a refined start/end
    # -- the video-side analog of what Whisper forced-alignment does for a
    # subtitle cue, narrowing (or, just as often, widening into the padding)
    # the coarse main-scan boundary to whatever the denser look actually
    # found. Falls back to the original coarse boundary if the dense re-scan
    # itself fails, or somehow finds nothing above threshold at all.
    refined: list[tuple[float, float, float | None]] = []
    # Retain dense raw observations alongside refined boundaries for exact
    # per-label policy evidence. Empty if refinement failed; coarse raw samples
    # are then used without fabricating semantic or body-part labels.
    all_window_scores: list[list] = []
    if candidates and on_stage is not None:
        await on_stage(f"Verifying {len(candidates)} candidate scene(s)")
    for candidate in candidates:
        start, end, frac, window_scores = await _refine_candidate_boundary(
            video_path,
            candidate.start,
            candidate.end,
            verify_pad,
            verify_max_pad,
            verify_interval,
            confidence_threshold,
            frame_concurrency,
        )
        if end - start < min_duration:
            # A single (or tightly clustered) dense hit can refine to a
            # near-zero-width span -- expand symmetrically around its midpoint
            # rather than storing a degenerate scene that's effectively
            # invisible to both the blur filter and the review UI's Start/End
            # fields.
            mid = (start + end) / 2
            start, end = max(0.0, mid - min_duration / 2), mid + min_duration / 2
        refined.append((start, end, frac))
        all_window_scores.append(window_scores)

    # Raw NudeNet class choices drive approval. The legacy semantic prompt
    # permits swimwear and must not veto explicitly selected body-part labels.
    staged_rows: list[DetectedScene] = []
    now = datetime.datetime.utcnow()
    auto_approved_count = need_review_count = 0
    for candidate, (start, end, frac), window_scores in zip(candidates, refined, all_window_scores):
        samples = window_scores or [s for s in scores if start <= s.timestamp <= end]
        evidence = nudenet_evidence(samples, confidence_threshold)
        auto_approve = auto_process and evidence is not None and not evidence["uncertain"]
        status = SceneReviewStatus.approved if auto_approve else SceneReviewStatus.pending
        auto_approved_count += int(status == SceneReviewStatus.approved)
        need_review_count += int(status == SceneReviewStatus.pending)
        staged_rows.append(
            DetectedScene(
                # Avoid SQLite rowid reuse: stale review forms must never
                # address a replacement row after a complete reset.
                id=secrets.randbits(62) + 1,
                title_id=title.id,
                scan_generation=job_id,
                start_seconds=start,
                end_seconds=end,
                peak_confidence=candidate.peak_confidence,
                verified_fraction=frac,
                category_evidence=json.dumps(evidence) if evidence is not None else None,
                detector_evidence=json.dumps({
                    "schema_version": 2, "backend": "nudenet", "semantic_coverage": [],
                    "samples": [{"timestamp": s.timestamp, "detections": s.detections}
                        for s in (window_scores or [s for s in scores if start <= s.timestamp <= end])
                        if s.detections is not None],
                }),
                mute_audio=False,
                status=status,
                reviewed_at=now if status != SceneReviewStatus.pending else None,
            )
        )
    current_policy = validate_policy(await get_setting(session, "scene_content_policy"))
    if selected != sorted(k for k in NUDENET_CATEGORIES if current_policy["actions"][k] != "allow"):
        raise ValueError("Detection label preferences changed during scanning; run a fresh scan")
    await _activate_scan(session, title.id, job_id, staged_rows, analysis={
        "schema_version": 1, "backend": "nudenet", "model": "NudeNet 3.4.2",
        # Only enabled labels proposed windows across the full title. Enabling
        # another label later requires a fresh scan, not invented coverage.
        "coverage": selected, "detector_categories": sorted(NUDENET_CATEGORIES),
        "frame_interval_seconds": frame_interval,
    })

    # Full success -- the checkpoint has served its purpose. A future scan
    # (even a manual rescan of this same title) gets a new job_id and thus a
    # fresh scan_work_dir regardless, so there's nothing to gain by keeping it.
    shutil.rmtree(scan_work_dir, ignore_errors=True)

    return ScanOutcome(candidate_count=len(candidates), auto_approved_count=auto_approved_count,
                       need_review_count=need_review_count)


async def _activate_scan(session: AsyncSession, title_id: int, job_id: int,
                         rows: list[DetectedScene], analysis: dict | None = None) -> None:
    """Swap a staged generation in one fenced transaction, with no callbacks.

    Job completion is part of activation, so replaying an already successful
    generation cannot overwrite later human review decisions. Interrupted jobs
    retain their identity and checkpoint but never partially visible scenes.
    """
    job = await session.get(SceneJob, job_id, populate_existing=True)
    fence = await session.execute(update(Title).where(
        Title.id == title_id, Title.scene_scan_generation == job_id,
        select(SceneJob.id).where(SceneJob.id == job_id,
            SceneJob.state.in_([JobState.queued, JobState.processing])).exists(),
    ).values(scene_scan_status="scanned" if rows else "no_scenes_found"))
    if not fence.rowcount:
        await session.rollback()
        raise ValueError("Scene scan was cancelled, completed or superseded; previous results retained")
    if analysis and analysis.get("backend") == "nudenet":
        # The activation write reservation also fences settings changes, so
        # newly enabled proposal classes cannot be missed by a late scan.
        current_policy = validate_policy(await get_setting(session, "scene_content_policy"))
        enabled = sorted(k for k in NUDENET_CATEGORIES if current_policy["actions"][k] != "allow")
        if enabled != analysis["coverage"]:
            await session.rollback()
            raise ValueError("Detection label preferences changed during scanning; previous scenes retained")
    conditions = [DetectedScene.title_id == title_id]
    if job.keep_manual:
        conditions.append(DetectedScene.source != "manual")
    previous = await session.execute(select(DetectedScene).where(*conditions))
    for scene in previous.scalars():
        await session.delete(scene)
    for row in rows:
        if row.id is None:
            row.id = secrets.randbits(62) + 1
    session.add_all(rows)
    title = await session.get(Title, title_id)
    title.scene_analysis = json.dumps(analysis or {"schema_version": 1, "backend": "legacy-exposure", "coverage": []})
    job.state = JobState.done
    job.finished_at = datetime.datetime.utcnow()
    await session.commit()


async def _scan_semantic(session, title, job_id, work_dir, fingerprint, duration,
                         on_progress, on_stage) -> ScanOutcome:
    """Broad opt-in proposals, independent of scalar exposure detections.

    Retain all observed labels, including currently allowed categories, so a
    later policy change can reuse evidence rather than repeat paid model calls.
    Failed windows abort the swap and preserve the last successful scene set.
    """
    if on_stage is not None:
        await on_stage("Sampling clothing/nudity across the whole title (external vision API)")
    windows = await scan_content_windows(
        video_path=Path(title.video_path), duration=duration,
        window_seconds=float(await get_setting(session, "scene_semantic_window_seconds")),
        checkpoint_dir=work_dir, fingerprint=fingerprint,
        base_url=await get_setting(session, "claude_vision_base_url"),
        api_key=await get_setting(session, "claude_vision_api_key"),
        model=await get_setting(session, "claude_vision_model"),
        ffmpeg_bin=app_settings.ffmpeg_bin, on_progress=on_progress,
    )
    auto_process = bool(await get_setting(session, "scene_auto_process"))
    policy = await get_setting(session, "scene_content_policy")
    rows = []
    approved = pending = 0
    for start, end, evidence in windows:
        if not evidence["labels"] and not evidence["uncertain"]:
            continue
        scene = DetectedScene(id=secrets.randbits(62) + 1, title_id=title.id,
            scan_generation=job_id, start_seconds=start, end_seconds=end,
            peak_confidence=max(evidence["labels"].values(), default=0.0),
            category_evidence=json.dumps(evidence), mute_audio=False,
            source="detected")
        decision = scene_decision(scene, policy)
        # Eligibility is NOT review status: allowed evidence can become blocking
        # later without losing observations or inventing a manual rejection.
        scene.status = (SceneReviewStatus.approved if auto_process and not decision.needs_review
                        else SceneReviewStatus.pending)
        if scene.status == SceneReviewStatus.approved:
            approved += 1
            scene.reviewed_at = datetime.datetime.utcnow()
        else:
            pending += 1
        rows.append(scene)
    await _activate_scan(session, title.id, job_id, rows, analysis={
        "schema_version": 1, "backend": "semantic-sampled-frames",
        "coverage": sorted(VISUAL_CATEGORIES), "windows": len(windows),
        "window_seconds": float(await get_setting(session, "scene_semantic_window_seconds")),
        "model": await get_setting(session, "claude_vision_model"),
        "scanned_at": datetime.datetime.utcnow().isoformat(),
    })
    shutil.rmtree(work_dir, ignore_errors=True)
    return ScanOutcome(len(rows), approved, pending)


async def apply_scene_blur(
    session: AsyncSession,
    title: Title,
    job_id: int | str,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
    output_path: Path | None = None,
    commit: bool = True,
    audio_source_path: Path | None = None,
) -> BlurOutcome:
    """Rebuild the "Vulgarr Edit" file from scratch, blurring+muting
    every currently-approved scene on this title -- not just newly-approved
    ones. The source file is never opened for writing -- a failed blur just
    means the temp file gets deleted, nothing about the original changes, and
    any previously-generated "Vulgarr Edit" file is left in place untouched.
    A configured output root selects a mirrored destination. When an audio
    master exists, reuse its clean tracks, never previously scene-muted audio.
    output_path pins the destination for a combined audio job; commit=False
    lets that caller commit its cache reference and scene state together.

    job_id keys apply_blur's persistent work_dir (app_settings.data_dir /
    "blur_work" / str(job_id)) -- on a container restart, scene_worker's own
    startup reconciliation resets an interrupted job back to "queued" and
    redispatches it with this *same* job_id, so the resumed attempt's
    work_dir naturally lines up with whatever the interrupted attempt already
    produced. A brand-new manual Apply click (a genuinely new job_id) always
    gets a fresh work_dir, never accidentally inherits a stale one.

    Deliberately *not* scoped to approved-and-unapplied: this always operates
    on the full current approved set so that rejecting a scene that was
    already baked into an earlier "Vulgarr Edit" -- the whole point of the
    "auto-process now, correct it after the fact" workflow (see
    scene_auto_process) -- and then re-running Apply actually regenerates the
    file *without* that scene, rather than silently leaving its old blur/mute
    in place forever. Each regeneration re-encodes fresh from the untouched
    original, so there's no generational quality loss from doing this
    repeatedly.

    Raises ValueError if there's nothing currently approved at all (the caller
    is expected to have already checked this before enqueuing a job, but it's
    re-checked here too since the set can change between "Apply" being
    clicked and the job actually running)."""
    result = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status == SceneReviewStatus.approved,
        )
    )
    scenes = result.scalars().all()
    policy = await get_setting(session, "scene_content_policy")
    render_settings = {key: await get_setting(session, key) for key in RENDER_SETTING_KEYS}
    set_revision = scene_set_fingerprint(scenes, policy, render_settings)
    requested_generation = getattr(title, "scene_scan_generation", None)
    decisions = {s.id: scene_decision(s, policy) for s in scenes}
    scenes = [s for s in scenes if decisions[s.id].action != Action.allow]
    # Empty effective sets deliberately rebuild from source/clean master,
    # removing previously applied blur/mute windows after policy changes.

    # Padded before merging (not after) -- a fixed safety margin against
    # boundary imprecision in an already-approved scene (a detected/adjusted
    # start or end that's a little tight against the real content, showing up
    # as a brief flash of unblurred/unmuted content right at the edge).
    # Padding two close-together approved scenes can make their windows
    # overlap; merging afterward (as already happened here regardless, for
    # independently-adjusted scenes) collapses that into one window rather
    # than erroring.
    #
    # Asymmetric on purpose (previously one symmetric value) -- direct
    # frame-by-frame ground truth found the classifier's boundary imprecision
    # is one-directional: it reliably catches a scene's real start, but
    # repeatedly stops firing several seconds before the shot actually cuts
    # away at the end (see scene_blur_pad_end_seconds's DEFAULT_SETTINGS
    # comment for the two real cases this was found on). No reason to pay for
    # a wide start pad the evidence doesn't call for.
    blur_pad_start = float(await get_setting(session, "scene_blur_pad_start_seconds"))
    blur_pad_end = float(await get_setting(session, "scene_blur_pad_end_seconds"))

    # Overlapping/adjacent approved windows (real after enough independent
    # "adjust" edits, or now from padding) are merged before building the
    # filter graph -- not required for correctness (ffmpeg's between()
    # OR-summation handles overlaps fine regardless), just keeps the term
    # count down against the batching limit. Every approved scene gets
    # blurred; only the subset with mute_audio set also gets its audio muted
    # -- a separate, explicit per-scene opt-in (default off) since
    # plot-relevant nudity doesn't always mean the dialogue over it needs
    # muting too. mute_scenes is always a subset of scenes.
    blur_merged = merge_intervals(
        [(max(0.0, s.start_seconds - blur_pad_start), s.end_seconds + blur_pad_end) for s in scenes],
        merge_gap_seconds=0.0,
    )
    blur_intervals = [MuteInterval(start=s, end=e) for s, e in blur_merged]

    mute_scenes = [s for s in scenes if decisions[s.id].mute_audio]
    mute_intervals: list[MuteInterval] = []
    if mute_scenes:
        mute_merged = merge_intervals(
            [(max(0.0, s.start_seconds - blur_pad_start), s.end_seconds + blur_pad_end) for s in mute_scenes],
            merge_gap_seconds=0.0,
        )
        mute_intervals = [MuteInterval(start=s, end=e) for s, e in mute_merged]

    video_crf = int(await get_setting(session, "blur_video_crf"))
    video_preset = await get_setting(session, "blur_video_preset")
    blur_radius = int(await get_setting(session, "scene_blur_radius"))
    blur_power = int(await get_setting(session, "scene_blur_power"))

    work_dir = app_settings.data_dir / "blur_work" / str(job_id)

    # build_blurred_video's on_progress only covers the video-blur phase --
    # the audio-mute split/re-encode/concat pipeline that runs afterward
    # (when mute_intervals is non-empty) only calls on_stage, which doesn't
    # move progress_percent at all. The caller (scene_worker) maps on_progress's
    # fraction=1.0 to a ~97% "almost done" ceiling that also gates whether a
    # job can still be cancelled -- left unscaled, that ceiling gets hit as
    # soon as video finishes, well before a long audio-mute phase (measured
    # up to ~27 minutes for a large multi-track file) actually completes,
    # making the job effectively uncancellable for that whole stretch. Halving
    # the reported fraction whenever there's an audio phase to follow keeps
    # progress (and the cancel floor built on top of it) from prematurely
    # implying the job is nearly done.
    video_progress = on_progress
    if on_progress is not None and mute_intervals:

        async def _scaled_progress(fraction: float, _inner=on_progress) -> None:
            await _inner(fraction * 0.5)

        video_progress = _scaled_progress

    from app.output import source_fingerprint, title_edit_destination
    output_path = output_path or await title_edit_destination(session, title, app_settings.media_root)
    output_options = {}
    if output_path is not None:
        output_options['output_path'] = output_path
        if audio_source_path is not None:
            # A combined audio job has validated a new master but must not
            # put its reference on Title yet: progress callbacks commit this
            # shared session and would expose pending state before publish.
            if not audio_source_path.is_file():
                raise ValueError('New clean audio master is missing')
            output_options['audio_source_path'] = audio_source_path
        elif title.clean_audio_path:
            audio_master = Path(title.clean_audio_path)
            if (not audio_master.is_file()
                    or title.clean_audio_source_fingerprint != source_fingerprint(Path(title.video_path))):
                raise ValueError('Cached clean audio is missing or the source changed. Reprocess audio before applying scenes.')
            output_options['audio_source_path'] = audio_master

    async def validate_current_set() -> None:
        """Reject stale queued/running Apply work before touching published media."""
        # Hold SQLite's writer reservation through publication and the caller's
        # commit. Review/settings writes cannot slip between validation and swap.
        lock = await session.execute(update(Title).where(
            Title.id == title.id, Title.scene_scan_generation == requested_generation,
        ).values(scene_scan_status=Title.scene_scan_status))
        if not lock.rowcount:
            raise ValueError("Scene generation changed during Apply; previous edit retained")
        current = (await session.execute(select(DetectedScene).where(
            DetectedScene.title_id == title.id,
        ).execution_options(populate_existing=True))).scalars().all()
        current_policy = await get_setting(session, "scene_content_policy")
        current_render_settings = {key: await get_setting(session, key) for key in RENDER_SETTING_KEYS}
        current_generation = (await session.execute(select(Title.scene_scan_generation).where(
            Title.id == title.id))).scalar_one()
        if (current_generation != requested_generation
                or policy_fingerprint(current_policy) != policy_fingerprint(policy)
                or scene_set_fingerprint(current, current_policy, current_render_settings) != set_revision):
            raise ValueError("Scene decisions or policy changed during Apply; previous published edit retained")

    final_path = await apply_blur(
        video_path=Path(title.video_path),
        **output_options,
        blur_intervals=blur_intervals,
        mute_intervals=mute_intervals,
        ffmpeg_bin=app_settings.ffmpeg_bin,
        ffprobe_bin=app_settings.ffprobe_bin,
        video_crf=video_crf,
        video_preset=video_preset,
        work_dir=work_dir,
        blur_radius=blur_radius,
        blur_power=blur_power,
        on_progress=video_progress,
        on_stage=on_stage,
        before_publish=validate_current_set,
    )

    now = datetime.datetime.utcnow()
    title.vulgarr_edit_path = str(final_path)
    title.vulgarr_edit_generated_at = now
    title.scene_policy_applied = policy_fingerprint(policy)
    title.scene_set_applied = set_revision
    for scene in scenes:
        scene.applied_at = now

    # A scene that was applied by an earlier run but isn't in this run's
    # approved set (rejected sometime after being baked in) is no longer
    # reflected in the file that was just written above -- clear its
    # applied_at so it doesn't keep reading as "currently applied" now that
    # it isn't.
    stale_applied = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.id.not_in([s.id for s in scenes]),
            DetectedScene.applied_at.is_not(None),
        )
    )
    for scene in stale_applied.scalars().all():
        scene.applied_at = None

    if commit:
        await session.commit()

    return BlurOutcome(output_path=final_path, scene_count=len(scenes))


async def reverify_scenes_with_claude(
    session: AsyncSession,
    title: Title,
    on_progress: ScanProgressCallback | None = None,
) -> ReverifyOutcome:
    """Re-runs the optional Claude Vision precision filter (app.vision.claude_verify)
    against every currently pending/approved DetectedScene on this title, using
    each scene's already-stored (and possibly hand-adjusted) start/end. The manual
    counterpart to the same check scan_for_scenes runs automatically during a fresh
    scan -- exists for the case that check's own settings docs warn about: scenes
    scanned (and, with scene_auto_process on, auto-approved) *before*
    claude_vision_verify_enabled was ever turned on never got this filter applied at
    all, and there's no "undo" for an auto-approval already made short of a human
    eyeballing every scene by hand.

    Only scenes not already rejected are touched -- a rejected candidate already has
    a verdict (human or Claude) and there's no upside to re-spending money confirming
    it again. Every remaining candidate is re-scored regardless of verified_fraction
    -- unlike a fresh scan, there's no claude_skip_above_fraction shortcut here, since
    that exists purely to skip *paying* for scenes NudeNet's own dense re-scan
    already trusted, which has no bearing on whether this manual recheck should run.

    A definite YES/NO verdict sets status accordingly (approved/rejected), the same
    semantics scan_for_scenes already uses -- an explicit Claude "no" is a considered
    verdict, not an unknown. A failed/unreachable call leaves the scene's existing
    status untouched (never downgrades an already-approved scene just because the
    endpoint hiccuped) but still records the failure in claude_verify_reason so it's
    visible which scenes still need an actual look.
    """
    semantic_enabled = bool(await get_setting(session, "scene_semantic_enabled"))
    legacy_enabled = bool(await get_setting(session, "claude_vision_verify_enabled"))
    if not semantic_enabled and not legacy_enabled:
        raise ValueError("External scene analysis is disabled; enable it explicitly in Settings")
    policy = await get_setting(session, "scene_content_policy")
    claude_base_url = await get_setting(session, "claude_vision_base_url")
    claude_api_key = await get_setting(session, "claude_vision_api_key")
    claude_model = await get_setting(session, "claude_vision_model")

    result = await session.execute(
        select(DetectedScene)
        .where(DetectedScene.title_id == title.id, DetectedScene.status != SceneReviewStatus.rejected)
        .order_by(DetectedScene.start_seconds)
    )
    scenes = result.scalars().all()

    approved_count = 0
    rejected_count = 0
    failed_count = 0
    now = datetime.datetime.utcnow()
    for i, scene in enumerate(scenes):
        if scene.category_evidence and set(json.loads(scene.category_evidence)["coverage"]) == NUDENET_CATEGORIES:
            # A legacy semantic verifier cannot reinterpret exact body-part
            # choices. Fresh local scans update these detector observations.
            approved_count += int(scene.status == SceneReviewStatus.approved)
            if on_progress is not None:
                await on_progress(i + 1, len(scenes))
            continue
        if scene.category_evidence or semantic_enabled:
            # Never feed structured clothing evidence back through a yes/no
            # nudity prompt (which explicitly allows swimwear).
            try:
                if not semantic_enabled:
                    raise ValueError("Structured analysis requires semantic opt-in")
                evidence = await classify_window(base_url=claude_base_url, api_key=claude_api_key,
                    model=claude_model, ffmpeg_bin=app_settings.ffmpeg_bin,
                    video_path=Path(title.video_path), start=scene.start_seconds, end=scene.end_seconds)
                scene.category_evidence = json.dumps(evidence)
                decision = scene_decision(scene, policy)
                if scene.source != "manual" and scene.action_override is None:
                    scene.status = SceneReviewStatus.pending if decision.needs_review else SceneReviewStatus.approved
                scene.reviewed_at = now
                approved_count += int(scene.status == SceneReviewStatus.approved)
            except ValueError:
                scene.claude_verify_reason = "Structured verification failed; evidence not replaced"
                failed_count += 1
            await session.commit()
            if on_progress is not None:
                await on_progress(i + 1, len(scenes))
            continue
        verify_result = await verify_candidate(
            base_url=claude_base_url,
            api_key=claude_api_key,
            model=claude_model,
            ffmpeg_bin=app_settings.ffmpeg_bin,
            video_path=Path(title.video_path),
            start=scene.start_seconds,
            end=scene.end_seconds,
        )
        if verify_result is not None:
            scene.mute_audio = verify_result.mute_audio
            reason = f"{'YES' if verify_result.confirmed else 'NO'}: {verify_result.reason}".strip(": ")
            if verify_result.mute_audio:
                reason += " (sex scene -- audio muted)"
            scene.claude_verify_reason = reason
            scene.status = SceneReviewStatus.approved if verify_result.confirmed else SceneReviewStatus.rejected
            scene.reviewed_at = now
            if verify_result.confirmed:
                approved_count += 1
            else:
                rejected_count += 1
        else:
            scene.claude_verify_reason = "Verification failed -- left for manual review"
            failed_count += 1
        await session.commit()
        if on_progress is not None:
            await on_progress(i + 1, len(scenes))

    return ReverifyOutcome(
        checked_count=len(scenes),
        approved_count=approved_count,
        rejected_count=rejected_count,
        failed_count=failed_count,
    )
