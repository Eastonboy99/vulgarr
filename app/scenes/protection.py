"""Opt-in, local-only protection-first automation. Sampled assurance, not a guarantee.

No Plex management. Only validated media is installed at the configured output.
Unknown content is conservatively filtered; failed inference is NOT a negative.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from sqlalchemy import select

from app.config import settings as app_settings
from app.db.models import DetectedScene, ProcessingJob, JobState, SceneReviewStatus
from app.db.session import get_setting
from app.output import source_fingerprint
from app.scenes.content_policy import Action, resolve_action, policy_fingerprint, validate_policy
from app.vision import contextual

VERSION = "protection-v1"
CONTEXT_WINDOW = 4.0
DETECTOR_INTERVAL = .25
BOUNDARY_SECONDS = 2.0
OUTPUT_RETRIES = 1
KEYS = ("scene_protection_enabled",)
SCAN_KEYS = ("scene_confidence_threshold", "scene_frame_interval_seconds", "scene_semantic_enabled",
             "scene_min_duration_seconds", "scene_merge_gap_seconds", "scene_verify_pad_seconds",
             "scene_verify_max_pad_seconds", "scene_verify_frame_interval_seconds",
             "scene_high_confidence_single_frame_threshold")


class ProtectionNotReady(ValueError):
    pass


class ResidualContentError(ValueError):
    def __init__(self, findings):
        super().__init__("Rendered output still contains policy-blocked or unresolved visible content; previous edit retained")
        self.findings = findings


async def enabled(session):
    return bool(await get_setting(session, "scene_protection_enabled"))


async def context_config(session):
    config = {key: await get_setting(session, key) for key in contextual.KEYS}
    if await enabled(session):
        config["scene_contextual_enabled"] = True
        config["scene_contextual_window_seconds"] = min(float(config["scene_contextual_window_seconds"]), CONTEXT_WINDOW)
    return config


async def fingerprint(session):
    snapshot = {key: await get_setting(session, key) for key in SCAN_KEYS}
    snapshot.update(version=VERSION, enabled=await enabled(session),
                    contextual=contextual.config_fingerprint(await context_config(session)),
                    detector_interval=DETECTOR_INTERVAL, boundary_seconds=BOUNDARY_SECONDS,
                    output_retries=OUTPUT_RETRIES,
                    policy=validate_policy(await get_setting(session, "scene_content_policy")))
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def blocked(evidence, policy):
    return resolve_action(policy, evidence, conservative=True).action != Action.allow


def validate_detector_coverage(scores, duration, interval):
    times = sorted({float(s.timestamp) for s in scores})
    if (not scores or any(s.detections is None for s in scores)
            or len(times) < max(1, round(duration / interval) - 1)
            or any(not math.isfinite(t) or t < 0 or t > duration for t in times)
            or times[0] > interval or duration - times[-1] > interval * 1.51
            or any(b - a > interval * 1.51 for a,b in zip(times, times[1:]))):
        raise ProtectionNotReady("Anatomy analysis returned incomplete full-title evidence; previous edit retained")


async def shot_windows(video_path, duration, ffmpeg_bin, *, local_gate_owned=False):
    """Full-title cut discovery; failure aborts instead of silently losing coverage."""
    from app.mux.remux import _run
    async def decode():
        return await _run([ffmpeg_bin, "-nostdin", "-v", "info", "-i", str(video_path),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "select='gt(scene,0.30)',metadata=print:file=-",
            "-f", "null", "-"], timeout=max(600, duration * 4))
    from app.vision.remote_source import remote_extract_enabled
    if remote_extract_enabled() and not local_gate_owned:
        from app.queue.heavy_pipeline_lock import HEAVY_PIPELINE_LOCK
        async with HEAVY_PIPELINE_LOCK:
            code, out, _err = await decode()
    else:
        code, out, _err = await decode()
    if code:
        raise ProtectionNotReady("Shot-change analysis failed; no new edit published")
    cuts = sorted(set(float(t) for t in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", out)))
    cuts = [t for t in cuts if math.isfinite(t) and 0 <= t < duration]
    # Include the first part of each new shot, even when shorter than the base cadence.
    return [(t, min(duration, t + .75, cuts[i + 1] if i + 1 < len(cuts) else duration))
            for i, t in enumerate(cuts) if min(duration, t + .75, cuts[i + 1] if i + 1 < len(cuts) else duration) > t]


async def analyze_windows(*, video_path, duration, checkpoint_dir, fingerprint, config,
                          policy, on_progress=None):
    shots = await shot_windows(video_path, duration, app_settings.ffmpeg_bin)
    common = dict(video_path=video_path, duration=duration, fingerprint=fingerprint,
        base_url=config["scene_contextual_base_url"], api_key=config["scene_contextual_api_key"],
        model=config["scene_contextual_model"], ffmpeg_bin=app_settings.ffmpeg_bin,
        window_seconds=config["scene_contextual_window_seconds"], timeout_seconds=config["scene_contextual_timeout_seconds"])
    async def discovery_progress(done,total):
        if on_progress and total: await on_progress(.6*done/total,1)
    windows = await contextual.scan_contextual_windows(**common, checkpoint_dir=checkpoint_dir/"discovery",
        extra_windows=shots, on_progress=discovery_progress)
    # One higher-resolution recheck of the SAME timestamps. Never veto a positive.
    resolved, checks = [], 0
    recheck_total = sum(e["uncertain"] for _,_,e in windows)
    for start, end, evidence in windows:
        if evidence["uncertain"]:
            checked = await contextual.scan_contextual_windows(**common,
                checkpoint_dir=checkpoint_dir/f"recheck-{start:.6f}-{end:.6f}",
                window_plan=[(start, end)], max_long_edge=1024, purpose="recheck")
            checks += 1
            if on_progress: await on_progress(.6+.2*checks/max(1,recheck_total),1)
            if not evidence["labels"]:
                evidence = checked[0][2]
        resolved.append((start, end, evidence))
    # Expand candidates outward with denser samples. Never narrow an original hit.
    boundaries = set()
    for start, end, evidence in resolved:
        if blocked(evidence, policy):
            if start > 0: boundaries.add((max(0, start - BOUNDARY_SECONDS), start))
            if end < duration: boundaries.add((end, min(duration, end + BOUNDARY_SECONDS)))
    async def boundary_progress(done,total):
        if on_progress and total: await on_progress(.8+.2*done/total,1)
    if boundaries:
        refined = await contextual.scan_contextual_windows(**common, checkpoint_dir=checkpoint_dir/"boundaries",
            window_plan=sorted(boundaries), purpose="recheck", on_progress=boundary_progress)
        resolved.extend(refined)
    if on_progress: await on_progress(1,1)
    return sorted(resolved, key=lambda w: (w[0], w[1])), {
        "base_windows": math.ceil(duration / common["window_seconds"]), "shot_windows": len(shots),
        "uncertainty_rechecks": checks, "boundary_checks": len(boundaries), "complete": True}


async def readiness(session, title):
    """Gate EVERY publication path, including no-scenes and combined audio jobs."""
    if not await enabled(session):
        return
    try:
        analysis = json.loads(title.scene_analysis or "{}")
        manifest = analysis["protection"]
        if (title.scene_scan_status not in ("scanned", "no_scenes_found")
                or manifest["fingerprint"] != await fingerprint(session)
                or manifest["source"] != source_fingerprint(Path(title.video_path))
                or manifest.get("generation") != title.scene_scan_generation
                or manifest.get("complete") is not True):
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ProtectionNotReady("Protection-first analysis is missing, failed, incomplete or stale; run a fresh scan. Previous edit retained") from None
    pending = await session.scalar(select(DetectedScene.id).where(DetectedScene.title_id == title.id,
        DetectedScene.status == SceneReviewStatus.pending).limit(1))
    if pending is not None:
        raise ProtectionNotReady("Unresolved manual/legacy scene decisions prevent automatic publication")


async def audio_ready(session, title):
    latest = await session.scalar(select(ProcessingJob).where(ProcessingJob.title_id == title.id)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc()).limit(1))
    if latest is not None and latest.state != JobState.done:
        return False
    if (title.last_processed_wordlist_version is not None
            and title.last_processed_wordlist_version < int(await get_setting(session, "wordlist_version"))):
        return False
    active = await session.scalar(select(ProcessingJob.id).where(ProcessingJob.title_id == title.id,
        ProcessingJob.state.in_([JobState.queued, JobState.processing])).limit(1))
    return (active is None and bool(title.clean_audio_path)
        and Path(title.clean_audio_path).is_file()
        and title.clean_audio_source_fingerprint == source_fingerprint(Path(title.video_path)))


async def output_findings(*, session, title, path, work_dir, policy, duration, ignored=(), covered=(), on_progress=None):
    """Offset full-title output samples plus shot starts; never trust input flags alone."""
    config = await context_config(session)
    step = config["scene_contextual_window_seconds"]
    # Different timestamps from source discovery. Explicitly include both edges.
    # Half a frame cadence, not half a window (which repeats source samples).
    offset = step / 8
    plan = [(0., min(duration, offset))]
    start = offset
    while start < duration:
        plan.append((start, min(duration, start + step)))
        start += step
    plan.extend(await shot_windows(path, duration, app_settings.ffmpeg_bin, local_gate_owned=True))
    # Raw anatomy checks are independent of the VLM's judgments.
    from app.vision.classifier import scan_video_frames
    from app.scenes.content_policy import nudenet_evidence
    async def anatomy_progress(done,total):
        if on_progress and total: await on_progress(.4*done/total,1)
    async def vision_progress(done,total):
        if on_progress and total: await on_progress(.4+.6*done/total,1)
    detector_offset = min(DETECTOR_INTERVAL / 2, duration / 2)
    scores = await scan_video_frames(path, app_settings.ffmpeg_bin, duration_seconds=duration-detector_offset,
        frame_interval_seconds=DETECTOR_INTERVAL, concurrency=int(await get_setting(session, "scene_frame_classify_concurrency")),
        local_gate_owned=True, start_offset=detector_offset, on_progress=anatomy_progress)
    validate_detector_coverage(scores, duration, DETECTOR_INTERVAL)
    findings = []
    threshold = float(await get_setting(session, "scene_confidence_threshold"))
    for score in scores:
        if score.detections is None:
            raise ProtectionNotReady("Output anatomy verification returned incomplete evidence")
        evidence = nudenet_evidence([score], threshold)
        if blocked(evidence, policy):
            findings.append((max(0, score.timestamp - DETECTOR_INTERVAL), min(duration, score.timestamp + DETECTOR_INTERVAL), evidence))
    windows = await contextual.scan_contextual_windows(video_path=path, duration=duration,
        checkpoint_dir=work_dir, fingerprint=source_fingerprint(path), window_plan=plan,
        base_url=config["scene_contextual_base_url"], api_key=config["scene_contextual_api_key"],
        model=config["scene_contextual_model"], ffmpeg_bin=app_settings.ffmpeg_bin,
        window_seconds=step, timeout_seconds=config["scene_contextual_timeout_seconds"], purpose="output", on_progress=vision_progress)
    for a,b,e in windows:
        if not blocked(e, policy):
            continue
        unknown_times = [a + (b-a)*contextual.FRACTIONS[int(r["frame_id"][1])]
            for r in e.get("observations", []) if r["state"] in contextual.AMBIGUOUS]
        # Expected obscuration is not a new failure. Positive discernible content
        # remains a failure even inside an existing blur window.
        if (not e["labels"] and unknown_times
                and all(any(left <= t <= right for left,right in covered) for t in unknown_times)):
            continue
        findings.append((a,b,e))
    # Honor explicit human exceptions, splitting rather than broad overlap vetoes.
    result = []
    for a,b,e in findings:
        pieces = [(a,b)]
        for left,right in ignored:
            pieces = [(x,y) for s,t in pieces for x,y in
                ([(s,t)] if right <= s or left >= t else [(s,min(t,left)),(max(s,right),t)]) if y > x]
        result.extend((a,b,e) for a,b in pieces)
    return result
