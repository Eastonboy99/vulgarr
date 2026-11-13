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
from app.scenes.content_policy import Action, VISUAL_CATEGORIES, NUDENET_CATEGORIES, CONTEXTUAL_CATEGORIES, RENDER_SETTING_KEYS, scene_decision, policy_fingerprint, scene_set_fingerprint, validate_policy, nudenet_evidence
from app.vision.nudenet_labels import NUDENET_LABELS, category_id
from app.vision.content_classifier import scan_content_windows, classify_window
from app.vision import contextual
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
        return await _scan_for_scenes(session, title, job_id, on_progress, on_stage, proposal_policy=policy)


async def _scan_for_scenes(
    session: AsyncSession,
    title: Title,
    job_id: int,
    on_progress: ScanProgressCallback | None = None,
    on_stage: StageCallback | None = None,
    *, proposal_policy: dict | None = None,
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
    from app.scenes import protection
    protected = await protection.enabled(session)
    protection_revision = await protection.fingerprint(session) if protected else None
    protection_source = protection.source_fingerprint(Path(title.video_path)) if protected else None
    if protected and bool(await get_setting(session, "scene_semantic_enabled")):
        raise ValueError("Protection-first mode requires NudeNet plus local contextual discovery, not legacy semantic analysis")
    contextual_config = await _contextual_config(session)
    if contextual_config["scene_contextual_enabled"]:
        contextual.validate_sampling(contextual_config["scene_contextual_window_seconds"], contextual_config["scene_contextual_timeout_seconds"])
        await contextual.local_endpoint(contextual_config["scene_contextual_base_url"])
    auto_process = protected or bool(await get_setting(session, "scene_auto_process"))
    confidence_threshold = float(await get_setting(session, "scene_confidence_threshold"))
    frame_interval = float(await get_setting(session, "scene_frame_interval_seconds"))
    if protected:
        frame_interval = min(frame_interval, protection.DETECTOR_INTERVAL)
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
    # Match the exact policy bound by detection_labels(), including across the
    # asynchronous local-endpoint preflight. Activation checks the latest policy.
    policy = validate_policy(proposal_policy if proposal_policy is not None else await get_setting(session, "scene_content_policy"))
    selected = sorted(k for k in NUDENET_CATEGORIES if policy["actions"][k] != "allow")
    fingerprint = _scan_fingerprint(video_path, frame_interval, duration)
    fingerprint += hashlib.sha256(repr(selected).encode()).hexdigest()[:16]
    if not fingerprint_path.exists() or fingerprint_path.read_text().strip() != fingerprint:
        resume_scores_path.unlink(missing_ok=True)
        fingerprint_path.write_text(fingerprint)

    if bool(await get_setting(session, "scene_semantic_enabled")):
        return await _scan_semantic(session, title, job_id, scan_work_dir,
                                    fingerprint, duration, on_progress, on_stage, contextual_config)

    scores = await scan_video_frames(
        video_path,
        app_settings.ffmpeg_bin,
        duration_seconds=duration,
        frame_interval_seconds=frame_interval,
        on_progress=_progress_slice(on_progress, 0, .35) if contextual_config["scene_contextual_enabled"] else on_progress,
        concurrency=frame_concurrency,
        resume_scores_path=resume_scores_path,
        on_status=on_stage,
    )

    if protected:
        protection.validate_detector_coverage(scores, duration, frame_interval)
    candidates = cluster_scenes(
        scores,
        confidence_threshold=confidence_threshold,
        frame_interval_seconds=frame_interval,
        merge_gap_seconds=merge_gap,
        high_confidence_override=confidence_threshold if protected else high_confidence_override,
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
        auto_approve = auto_process and evidence is not None and (protected or not evidence["uncertain"])
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
                    "protection_conservative": protected, "automatic_decision": auto_approve,
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
    contextual_analysis = await _add_contextual_rows(session, title, job_id, staged_rows, scan_work_dir,
        fingerprint, duration, contextual_config, on_progress, on_stage)
    contextual_analysis.pop("added_rows")
    if protected:
        contextual_analysis["protection"] = {"fingerprint": protection_revision,
            "source": protection_source, "generation": job_id, "complete": True}
    auto_approved_count = sum(r.status == SceneReviewStatus.approved for r in staged_rows)
    need_review_count = sum(r.status == SceneReviewStatus.pending for r in staged_rows)
    await _activate_scan(session, title.id, job_id, staged_rows, analysis={
        "schema_version": 1, "backend": "nudenet", "model": "NudeNet 3.4.2",
        **contextual_analysis,
        # Only enabled labels proposed windows across the full title. Enabling
        # another label later requires a fresh scan, not invented coverage.
        "coverage": sorted(set(selected) | (CONTEXTUAL_CATEGORIES if contextual_config["scene_contextual_enabled"] else set())), "detector_categories": sorted(NUDENET_CATEGORIES),
        "frame_interval_seconds": frame_interval,
    })

    # Full success -- the checkpoint has served its purpose. A future scan
    # (even a manual rescan of this same title) gets a new job_id and thus a
    # fresh scan_work_dir regardless, so there's nothing to gain by keeping it.
    shutil.rmtree(scan_work_dir, ignore_errors=True)

    return ScanOutcome(candidate_count=len(staged_rows), auto_approved_count=auto_approved_count,
                       need_review_count=need_review_count)


def _progress_slice(callback, start, end):
    if callback is None:
        return None
    async def progress(done, total):
        fraction = min(1, max(0, done / total)) if total > 0 else 0
        await callback(round((start + (end - start) * fraction) * 10000), 10000)
    return progress


async def _contextual_config(session):
    from app.scenes.protection import context_config
    return await context_config(session)


async def _add_contextual_rows(session, title, job_id, rows, work_dir, fingerprint,
                               duration, config, on_progress, on_stage):
    """Stage independent proposals without publishing or vetoing NudeNet rows.

    Adjacent equivalent windows coalesce. Subtract already-blocking detector
    intervals so overlapping analyzers do not create duplicate review scenes.
    No NudeNet refinement is applied: it cannot validate obscured nudity.
    """
    analysis = {"contextual_fingerprint": contextual.config_fingerprint(config),
                "contextual_enabled": bool(config["scene_contextual_enabled"]), "added_rows": 0}
    if not config["scene_contextual_enabled"]:
        return analysis
    if on_stage:
        await on_stage("Independent local contextual-nudity discovery across the whole title (review required)")
    from app.scenes import protection
    protected = await protection.enabled(session)
    if protected:
        if on_stage:
            await on_stage("Protection-first discovery, shot checks and conservative local rechecks")
        windows, coverage = await protection.analyze_windows(video_path=Path(title.video_path), duration=duration,
            checkpoint_dir=work_dir / "protection", fingerprint=fingerprint, config=config,
            policy=await get_setting(session, "scene_content_policy"), on_progress=_progress_slice(on_progress, .35, .99))
        analysis["protection_coverage"] = coverage
    else:
        windows = await contextual.scan_contextual_windows(video_path=Path(title.video_path), duration=duration,
        checkpoint_dir=work_dir, fingerprint=fingerprint,
        base_url=config["scene_contextual_base_url"], api_key=config["scene_contextual_api_key"],
        model=config["scene_contextual_model"], ffmpeg_bin=app_settings.ffmpeg_bin,
        window_seconds=config["scene_contextual_window_seconds"],
        timeout_seconds=config["scene_contextual_timeout_seconds"], on_progress=_progress_slice(on_progress, .35, .99))
    policy = await get_setting(session, "scene_content_policy")
    blockers = [(r.start_seconds, r.end_seconds, scene_decision(r, policy).action) for r in rows]
    groups = []
    for start, end, evidence in windows:
        if not evidence["labels"] and not evidence["uncertain"]:
            continue
        if (groups and start <= groups[-1][1] + 1e-6
                and groups[-1][2]["labels"] == evidence["labels"]
                and groups[-1][2]["uncertain"] == evidence["uncertain"]):
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][3] += 1
            groups[-1][4].append({"start": start, "end": end, "evidence": evidence})
        else:
            groups.append([start, end, dict(evidence), 1, [{"start": start, "end": end, "evidence": evidence}]])
    added = 0
    for start, end, evidence, count, observations in groups:
        segments = [(start, end)]
        from app.scenes.content_policy import resolve_action
        requested_action = resolve_action(policy, evidence, conservative=protected).action
        for left, right, blocking_action in blockers:
            if (blocking_action == Action.allow or requested_action == Action.allow
                    or requested_action == Action.blur_mute and blocking_action != Action.blur_mute):
                continue
            remainder = []
            for a, b in segments:
                if right <= a or left >= b:
                    remainder.append((a, b))
                else:
                    if a < left:
                        remainder.append((a, left))
                    if right < b:
                        remainder.append((right, b))
            segments = remainder
        evidence["explanation"] = (f"Local AI candidate across {count} sampled window(s); " +
            ("protection-first automatic decision. " if protected else "human review required. ") + evidence["explanation"])[:1000]
        for a, b in segments:
            if b - a < 1e-6:
                continue
            rows.append(DetectedScene(id=secrets.randbits(62) + 1, title_id=title.id,
                scan_generation=job_id, start_seconds=a, end_seconds=b,
                peak_confidence=1.0 if evidence["labels"] else 0.0,
                verified_fraction=None, category_evidence=json.dumps(evidence),
                detector_evidence=json.dumps({"backend": "local-contextual", "version": contextual.VERSION,
                    "window_seconds": config["scene_contextual_window_seconds"],
                    "fractions": contextual.FRACTIONS, "window_count": count, "windows": observations,
                    "protection_conservative": protected, "automatic_decision": protected}),
                mute_audio=False, status=SceneReviewStatus.approved if protected else SceneReviewStatus.pending,
                reviewed_at=datetime.datetime.utcnow() if protected else None, source="detected"))
            added += 1
    analysis.update(added_rows=added, contextual={"model": config["scene_contextual_model"],
        "windows": len(windows), "window_seconds": config["scene_contextual_window_seconds"],
        "samples_per_window": len(contextual.FRACTIONS), "review_required": not protected})
    return analysis


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
    if analysis and "contextual_fingerprint" in analysis:
        if contextual.config_fingerprint(await _contextual_config(session)) != analysis["contextual_fingerprint"]:
            await session.rollback()
            raise ValueError("Contextual discovery settings changed during scanning; previous scenes retained")
    if analysis and analysis.get("backend") == "nudenet":
        # The activation write reservation also fences settings changes, so
        # newly enabled proposal classes cannot be missed by a late scan.
        current_policy = validate_policy(await get_setting(session, "scene_content_policy"))
        enabled = sorted(k for k in NUDENET_CATEGORIES if current_policy["actions"][k] != "allow")
        if enabled != sorted(set(analysis["coverage"]) & NUDENET_CATEGORIES):
            await session.rollback()
            raise ValueError("Detection label preferences changed during scanning; previous scenes retained")
    from app.scenes import protection
    if bool(analysis and "protection" in analysis) != await protection.enabled(session):
        await session.rollback()
        raise ValueError("Protection-first mode changed during scanning; previous scenes retained")
    if analysis and "protection" in analysis:
        if (analysis["protection"]["fingerprint"] != await protection.fingerprint(session)
                or analysis["protection"]["source"] != protection.source_fingerprint(Path((await session.get(Title, title_id)).video_path))):
            await session.rollback()
            raise ValueError("Protection-first policy, source or sampling changed during scanning; previous scenes retained")
    conditions = [DetectedScene.title_id == title_id]
    if job.keep_manual:
        conditions.append(DetectedScene.source != "manual")
    previous = await session.execute(select(DetectedScene).where(*conditions))
    for scene in previous.scalars():
        try:
            provenance = json.loads(scene.detector_evidence or "{}")
        except (ValueError, TypeError):
            provenance = {}
        if not isinstance(provenance, dict):
            provenance = {}
        if analysis and "protection" in analysis and (scene.source == "manual" or scene.action_override is not None
                or scene.audio_override is not None or scene.status == SceneReviewStatus.rejected
                or provenance.get("human_reviewed") is True
                or (scene.status == SceneReviewStatus.approved and not provenance.get("automatic_decision"))):
            # Keep explicit human intent across automated rescans. Rendering/output
            # checks honor rejected/allowed intervals instead of overriding them.
            continue
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
                         on_progress, on_stage, contextual_config=None) -> ScanOutcome:
    """Broad opt-in proposals, independent of scalar exposure detections.

    Retain all observed labels, including currently allowed categories, so a
    later policy change can reuse evidence rather than repeat paid model calls.
    Failed windows abort the swap and preserve the last successful scene set.
    """
    contextual_config = contextual_config or await _contextual_config(session)
    if on_stage is not None:
        await on_stage("Sampling clothing/nudity across the whole title (configured vision endpoint)")
    windows = await scan_content_windows(
        video_path=Path(title.video_path), duration=duration,
        window_seconds=float(await get_setting(session, "scene_semantic_window_seconds")),
        checkpoint_dir=work_dir, fingerprint=fingerprint,
        base_url=await get_setting(session, "claude_vision_base_url"),
        api_key=await get_setting(session, "claude_vision_api_key"),
        model=await get_setting(session, "claude_vision_model"),
        ffmpeg_bin=app_settings.ffmpeg_bin,
        on_progress=_progress_slice(on_progress, 0, .35) if contextual_config["scene_contextual_enabled"] else on_progress,
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
    contextual_analysis = await _add_contextual_rows(session, title, job_id, rows, work_dir,
        fingerprint, duration, contextual_config, on_progress, on_stage)
    pending += contextual_analysis.pop("added_rows")
    await _activate_scan(session, title.id, job_id, rows, analysis={
        "schema_version": 1, "backend": "semantic-sampled-frames", **contextual_analysis,
        "coverage": sorted(VISUAL_CATEGORIES | (CONTEXTUAL_CATEGORIES if contextual_config["scene_contextual_enabled"] else set())), "windows": len(windows),
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
    from app.scenes import protection
    protected = await protection.enabled(session)
    if protected:
        await session.refresh(title)
        await protection.readiness(session, title)
        if audio_source_path is None and not await protection.audio_ready(session, title):
            raise protection.ProtectionNotReady("Clean audio is not ready or an audio job is active; publication deferred")
    policy = await get_setting(session, "scene_content_policy")
    render_settings = {key: await get_setting(session, key) for key in RENDER_SETTING_KEYS}
    set_revision = scene_set_fingerprint(scenes, policy, render_settings)
    requested_generation = getattr(title, "scene_scan_generation", None)
    protection_revision = await protection.fingerprint(session) if protected else None
    def exceptions(all_scenes):
        return sorted((s.start_seconds, s.end_seconds) for s in all_scenes
            if s.status == SceneReviewStatus.rejected or s.action_override == "allow")
    ignored = exceptions((await session.execute(select(DetectedScene).where(DetectedScene.title_id == title.id))).scalars().all()) if protected else []
    mute_exceptions = sorted((s.start_seconds, s.end_seconds) for s in (await session.execute(select(DetectedScene).where(DetectedScene.title_id == title.id))).scalars().all()
        if s.audio_override is False or (s.action_override == "blur" and s.audio_override is not True)) if protected else []
    def subtract(windows, excluded=None):
        pieces = [(w.start, w.end) for w in windows]
        for a,b in ignored if excluded is None else excluded:
            pieces = [(x,y) for s,t in pieces for x,y in
                ([(s,t)] if b <= s or a >= t else [(s,min(t,a)),(max(s,b),t)]) if y > x]
        return [MuteInterval(start=a,end=b) for a,b in pieces]
    all_approved_scenes = list(scenes)
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

    if protected:
        blur_intervals, mute_intervals = subtract(blur_intervals), subtract(mute_intervals, ignored + mute_exceptions)
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

    if protected and on_progress is not None:
        async def protected_video_progress(fraction):
            await on_progress(fraction * (.45 if mute_intervals else .6))
        video_progress = protected_video_progress

    from app.output import source_fingerprint, title_edit_destination
    output_path = output_path or await title_edit_destination(session, title, app_settings.media_root)
    output_options = {}
    if protected and output_path is None:
        raise protection.ProtectionNotReady("Protection-first mode requires a configured clean output folder for this title")
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
        if await protection.enabled(session) != protected:
            raise ValueError("Protection-first mode changed during Apply; previous edit retained")
        if protected and isinstance(job_id, int):
            active_job = await session.get(SceneJob, job_id, populate_existing=True)
            if active_job is not None and active_job.state not in (JobState.queued, JobState.processing):
                raise ValueError("Apply was cancelled or completed; previous edit retained")
        current = (await session.execute(select(DetectedScene).where(
            DetectedScene.title_id == title.id,
        ).execution_options(populate_existing=True))).scalars().all()
        if protected:
            await session.refresh(title)
            await protection.readiness(session, title)
            if (protection_revision != await protection.fingerprint(session) or exceptions(current) != ignored
                    or sorted((s.start_seconds, s.end_seconds) for s in current if s.audio_override is False or (s.action_override == "blur" and s.audio_override is not True)) != mute_exceptions):
                raise protection.ProtectionNotReady("Protection-first configuration or manual exceptions changed during rendering")
        current_policy = await get_setting(session, "scene_content_policy")
        current_render_settings = {key: await get_setting(session, key) for key in RENDER_SETTING_KEYS}
        current_generation = (await session.execute(select(Title.scene_scan_generation).where(
            Title.id == title.id))).scalar_one()
        if (current_generation != requested_generation
                or policy_fingerprint(current_policy) != policy_fingerprint(policy)
                or scene_set_fingerprint(current, current_policy, current_render_settings) != set_revision):
            raise ValueError("Scene decisions or policy changed during Apply; previous published edit retained")

    async def verification_progress(done,total):
        if on_progress is not None and total:
            await on_progress(.65 + .3*done/total)

    async def validate_output(path):
        findings = await protection.output_findings(session=session, title=title, path=path,
            work_dir=work_dir / "output-check", policy=policy,
            duration=float((await probe(app_settings.ffprobe_bin, Path(title.video_path)))["format"]["duration"]),
            ignored=ignored, covered=[(w.start,w.end) for w in blur_intervals], on_progress=verification_progress)
        if findings:
            raise protection.ResidualContentError(findings)

    for attempt in range(protection.OUTPUT_RETRIES + 1 if protected else 1):
        try:
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
                **({"validate_output": validate_output} if protected else {}),
            )
            break
        except protection.ResidualContentError as exc:
            if attempt >= protection.OUTPUT_RETRIES:
                raise
            # Fence user edits BEFORE recording/rendering additional automatic coverage.
            await validate_current_set()
            now = datetime.datetime.utcnow()
            for a,b,evidence in exc.findings:
                if any(w.start <= a and w.end >= b for w in blur_intervals):
                    continue  # Existing coverage needs stronger blur, not duplicate rows.
                scene = DetectedScene(id=secrets.randbits(62)+1, title_id=title.id,
                    scan_generation=requested_generation, start_seconds=a, end_seconds=b,
                    peak_confidence=max(evidence["labels"].values(), default=0),
                    category_evidence=json.dumps(evidence), detector_evidence=json.dumps({
                        "backend": "local-output-correction", "automatic_decision": True, "protection_conservative": True}),
                    status=SceneReviewStatus.approved, reviewed_at=now, mute_audio=False, source="detected")
                session.add(scene)
                scenes.append(scene)
                all_approved_scenes.append(scene)
                action = scene_decision(scene, policy)
                blur_intervals.append(MuteInterval(start=max(0,a-blur_pad_start),end=b+blur_pad_end))
                if action.mute_audio:
                    mute_intervals.append(MuteInterval(start=max(0,a-blur_pad_start),end=b+blur_pad_end))
            blur_intervals = subtract([MuteInterval(start=a,end=b) for a,b in merge_intervals([(w.start,w.end) for w in blur_intervals], merge_gap_seconds=0)])
            mute_intervals = subtract([MuteInterval(start=a,end=b) for a,b in merge_intervals([(w.start,w.end) for w in mute_intervals], merge_gap_seconds=0)], ignored + mute_exceptions)
            set_revision = scene_set_fingerprint(all_approved_scenes, policy, render_settings)
            await session.commit()
            blur_radius, blur_power = max(90,blur_radius), max(8,blur_power)
            # Reuse neither old rendered segments nor their weaker blur parameters.
            work_dir = work_dir / "correction"
    now = datetime.datetime.utcnow()
    title.vulgarr_edit_path = str(final_path)
    title.vulgarr_edit_generated_at = now
    title.scene_policy_applied = policy_fingerprint(policy)
    title.scene_set_applied = set_revision
    if protected:
        analysis = json.loads(title.scene_analysis)
        analysis["protection"]["output_validation"] = {"passed": True, "attempts": attempt + 1,
            "blur_radius": blur_radius, "blur_power": blur_power, "checked_at": now.isoformat()}
        title.scene_analysis = json.dumps(analysis)
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
    if protected:
        shutil.rmtree(app_settings.data_dir / "blur_work" / str(job_id), ignore_errors=True)

    return BlurOutcome(output_path=final_path, scene_count=len(scenes))


async def reverify_scenes_with_claude(
    session: AsyncSession,
    title: Title,
    on_progress: ScanProgressCallback | None = None,
) -> ReverifyOutcome:
    """Recheck existing non-rejected legacy windows at the configured endpoint.

    This is not whole-title discovery. Exact NudeNet and local contextual
    evidence are skipped: a generic yes/no prompt must not veto those distinct
    observations. Legacy structured evidence uses its own semantic opt-in.
    The historical function/setting names remain for compatibility, but the
    endpoint can be local (LM Studio) or external; the UI explains the difference.

    A definite legacy YES/NO may approve/reject an eligible legacy candidate.
    A failed call retains its previous status and records only a safe failure
    marker. Existing published media is never modified by this operation.
    """
    if bool(await get_setting(session, "scene_protection_enabled")):
        raise ValueError("Protection-first mode uses its own local rechecks; legacy candidate verification is disabled")
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
        if scene.category_evidence and set(json.loads(scene.category_evidence)["coverage"]) in (NUDENET_CATEGORIES, CONTEXTUAL_CATEGORIES):
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
