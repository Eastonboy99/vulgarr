import asyncio
import datetime
import json
import secrets
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, update

from app.config import settings as app_settings
from app.db.models import DetectedScene, SceneJob, Title
from app.db.session import get_session, get_setting
from app.domain import SceneJobKind, SceneReviewStatus
from app.queue.scene_worker import scene_job_queue
from app.scenes import scrub_proxy
from app.scenes.content_policy import Action, CATEGORIES, VISUAL_CATEGORIES, RENDER_SETTING_KEYS, scene_decision, policy_fingerprint, scene_set_fingerprint
from app.vision.classifier import extract_frame

router = APIRouter(tags=["scenes"])
templates = Jinja2Templates(directory="app/templates")


async def _scene_review_context(session, title_id: int) -> dict:
    title = await session.get(Title, title_id)
    scenes_result = await session.execute(
        select(DetectedScene).where(DetectedScene.title_id == title_id).order_by(DetectedScene.start_seconds)
    )
    scenes = scenes_result.scalars().all()
    policy = await get_setting(session, "scene_content_policy")
    render_settings = {key: await get_setting(session, key) for key in RENDER_SETTING_KEYS}
    category_counts = {}
    for scene in scenes:
        scene.effective_decision = scene_decision(scene, policy)
        evidence = json.loads(scene.category_evidence) if scene.category_evidence else None
        scene.evidence_display = evidence
        if evidence:
            for category in evidence["labels"]:
                category_counts[category] = category_counts.get(category, 0) + 1
    high_confidence_fraction = float(await get_setting(session, "scene_high_confidence_fraction"))

    # The most recent job for this title, whatever its state -- while it's still
    # queued/processing this is what drives the real progress%/message shown
    # instead of a bare "Scanning…" spinner; once done it's still useful context
    # (e.g. a failed job's error message) even though scene_scan_status alone
    # already reflects the outcome.
    scan_job = (
        await session.execute(
            select(SceneJob)
            .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.scan)
            .order_by(SceneJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    # Same reasoning as scan_job above, for the blur/Apply side.
    blur_job = (
        await session.execute(
            select(SceneJob)
            .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.blur)
            .order_by(SceneJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    # Same reasoning again, for the "Reprocess with Claude Vision" action --
    # see reverify_scenes_with_claude.
    claude_verify_job = (
        await session.execute(
            select(SceneJob)
            .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.claude_verify)
            .order_by(SceneJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    pending_count = sum(1 for s in scenes if s.status == SceneReviewStatus.pending)
    high_confidence_pending_count = sum(
        1
        for s in scenes
        if s.status == SceneReviewStatus.pending
        and s.verified_fraction is not None
        and s.verified_fraction >= high_confidence_fraction
    )
    # What a "Reprocess with Claude Vision" click would actually touch -- see
    # reverify_scenes_with_claude, which skips already-rejected scenes.
    claude_reverifiable_count = sum(1 for s in scenes if s.status != SceneReviewStatus.rejected)

    return {
        "title_id": title_id,
        "scan_generation": str(title.scene_scan_generation or "") if title else "",
        "category_counts": category_counts,
        "category_labels": CATEGORIES,
        "policy_stale": bool(title and title.vulgarr_edit_path and title.scene_policy_applied != policy_fingerprint(policy)),
        "scene_set_stale": bool(title and title.vulgarr_edit_path and title.scene_set_applied != scene_set_fingerprint(scenes, policy, render_settings)),
        "scene_analysis": json.loads(title.scene_analysis) if title and title.scene_analysis else None,
        "needs_reanalysis": not (title and title.scene_analysis and set(json.loads(title.scene_analysis).get("coverage", [])) >= VISUAL_CATEGORIES),
        "legacy_evidence": any(not s.category_evidence and s.source != "manual" for s in scenes),
        "scene_scan_status": title.scene_scan_status if title else "not_scanned",
        "scenes": scenes,
        "scan_job": scan_job,
        "blur_job": blur_job,
        "claude_verify_job": claude_verify_job,
        "claude_vision_verify_enabled": bool(await get_setting(session, "claude_vision_verify_enabled")) or bool(await get_setting(session, "scene_semantic_enabled")),
        "claude_reverifiable_count": claude_reverifiable_count,
        "vulgarr_edit_path": title.vulgarr_edit_path if title else None,
        "high_confidence_fraction": high_confidence_fraction,
        "pending_count": pending_count,
        "high_confidence_pending_count": high_confidence_pending_count,
    }


async def _claim_review_generation(session, request: Request, title_id: int) -> bool:
    """Fence title-wide stale forms and reserve the writer through review commit."""
    form = await request.form()
    raw = form.get("scan_generation", "")
    try:
        generation = int(raw) if raw else None
    except (ValueError, TypeError):
        return False
    claimed = await session.execute(update(Title).where(
        Title.id == title_id, Title.scene_scan_generation == generation,
        Title.scene_scan_status.not_in(["queued", "scanning"]),
    ).values(scene_scan_status=Title.scene_scan_status))
    return bool(claimed.rowcount)


async def _render_scene_review(request: Request, title_id: int, detail_view: bool = False) -> HTMLResponse:
    """Scene-review actions are reachable both from the standalone scene-review
    page/section (just this fragment) and embedded on the title detail page
    (where the Approve-all/Apply row now lives up in the Video column, outside
    this fragment's own swap boundary) -- detail_view mirrors the same flag
    library.py's _render_title already uses for the same reason, re-rendering
    the whole card instead of just this section so nothing outside it goes
    stale after an action."""
    if detail_view:
        from app.routers.library import (
            _load_last_job,
            _load_last_scan_job,
            _load_matched_cues,
            _pending_scene_title_ids,
            _row_dict,
        )

        async with get_session() as session:
            current_version = int(await get_setting(session, "wordlist_version"))
            title = await session.get(Title, title_id)
            if title is None:
                return HTMLResponse("")
            pending_ids = await _pending_scene_title_ids(session, [title_id])
            row = _row_dict(title, current_version, has_pending_scenes=title_id in pending_ids)
            context = await _scene_review_context(session, title_id)
        last_job, last_job_duration = await _load_last_job(title_id)
        last_scan_job, last_scan_job_duration = await _load_last_scan_job(title_id)
        matched_cues = await _load_matched_cues(title_id)
        return templates.TemplateResponse(
            "partials/title_detail_card.html",
            {
                "request": request,
                "row": row,
                "last_job": last_job,
                "last_job_duration": last_job_duration,
                "last_scan_job": last_scan_job,
                "last_scan_job_duration": last_scan_job_duration,
                "matched_cues": matched_cues,
                **context,
            },
        )
    async with get_session() as session:
        context = await _scene_review_context(session, title_id)
    return templates.TemplateResponse(
        "partials/scene_review_list.html", {"request": request, **context}
    )


@router.get("/library/title/{title_id}/scenes", response_class=HTMLResponse)
async def scene_review_refresh(request: Request, title_id: int, detail_view: bool = False):
    """Self-polling target for scene_review_list.html while a scan is in progress
    (see its own hx-get) -- same idea as title_row.html's self-poll on
    queued/processing status."""
    return await _render_scene_review(request, title_id, detail_view)


@router.get("/library/title/{title_id}/scene-review", response_class=HTMLResponse)
async def scene_review_page(request: Request, title_id: int):
    """Standalone scene-review page. Every title (movie or episode) also gets
    this same list embedded inline on its own detail page
    (title_detail_card.html, reachable via title_href() for both media
    types) -- this route exists as a quicker jump straight to just this
    section, e.g. from title_row.html's "Review" link, without opening the
    full detail page."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is None:
            raise HTTPException(status_code=404)
        context = await _scene_review_context(session, title_id)
    return templates.TemplateResponse(
        "scene_review_page.html", {"request": request, "title": title, **context}
    )


@router.post("/library/title/{title_id}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes(request: Request, title_id: int, detail_view: bool = Form(False),
                      keep_manual: bool = Form(False)):
    await scene_job_queue.enqueue_scan(title_id, keep_manual=keep_manual)
    return await _render_scene_review(request, title_id, detail_view)


@router.post("/library/title/{title_id}/reverify-scenes-claude", response_class=HTMLResponse)
async def reverify_scenes_claude(request: Request, title_id: int, detail_view: bool = Form(False)):
    """Re-analyze existing windows with explicit external-analysis consent.

    Structured clothing evidence never goes through the legacy yes/no prompt.
    This is not a replacement for a whole-title scan when coverage is missing.
    """
    async with get_session() as session:
        if not (await get_setting(session, "claude_vision_verify_enabled") or await get_setting(session, "scene_semantic_enabled")):
            raise HTTPException(409, "External scene analysis must be explicitly enabled in Settings")
    await scene_job_queue.enqueue_claude_verify_if_not_already_active(title_id)
    return await _render_scene_review(request, title_id, detail_view)


@router.post("/library/title/{title_id}/apply-scenes", response_class=HTMLResponse)
async def apply_scenes(request: Request, title_id: int, detail_view: bool = Form(False)):
    """Rebuilds the "Vulgarr Edit" sibling file from every currently-approved
    scene on this title. Real work happens in the queue worker
    (apply_scene_blur via SceneJobKind.blur) -- this just enqueues, same shape
    as scan-scenes above.

    Gated on "any approved scene exists" rather than "approved and unapplied"
    -- this is also the manual trigger for the "reject a scene that was
    already baked into an earlier Vulgarr Edit, then re-process" flow (see
    apply_scene_blur), where every approved scene may already carry an
    applied_at from the prior run but the file still needs regenerating to
    drop the one that's no longer approved."""
    async with get_session() as session:
        if not await _claim_review_generation(session, request, title_id):
            return await _render_scene_review(request, title_id, detail_view)
        approved = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.approved,
            )
        )
        has_approved = approved.first() is not None
        title = await session.get(Title, title_id)
        has_published = bool(title and title.vulgarr_edit_path)
    if has_approved or has_published:
        await scene_job_queue.enqueue_blur_if_not_already_active(title_id)
    return await _render_scene_review(request, title_id, detail_view)


@router.post("/library/shows/{series_id}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes_show(request: Request, series_id: int, came_from: str | None = Form(default=None)):
    from app.db.models import MediaType
    from app.routers.library import SEVERITY_OPTIONS, _aggregate_show_pip, _load_seasons

    async with get_session() as session:
        result = await session.execute(
            select(Title.id).where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
        )
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=came_from
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    queued = sum([await scene_job_queue.enqueue_scan_if_not_already_active(title_id) for title_id in title_ids])
    scan_message = f"Queued {queued} episode(s) for scene scanning."

    return templates.TemplateResponse(
        "partials/show_detail_card.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": came_from,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
            "scan_message": scan_message,
        },
    )


@router.post("/library/shows/{series_id}/season/{season_number}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes_season(
    request: Request,
    series_id: int,
    season_number: int,
    detail_view: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    from app.db.models import MediaType
    from app.routers.library import SEVERITY_OPTIONS, _load_season, _load_season_detail_context

    async with get_session() as session:
        result = await session.execute(
            select(Title.id).where(
                Title.sonarr_series_id == series_id,
                Title.season_number == season_number,
                Title.media_type == MediaType.episode,
            )
        )
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        if detail_view:
            context = await _load_season_detail_context(
                session, series_id, season_number, current_version, came_from=came_from
            )
        else:
            season = await _load_season(session, series_id, season_number, current_version)

    queued = sum([await scene_job_queue.enqueue_scan_if_not_already_active(title_id) for title_id in title_ids])
    scan_message = f"Queued {queued} episode(s) for scene scanning."

    if detail_view:
        return templates.TemplateResponse(
            "partials/season_detail_card.html",
            {
                "request": request,
                "series_id": series_id,
                "season_number": season_number,
                "severity_options": SEVERITY_OPTIONS,
                "scan_message": scan_message,
                **context,
            },
        )
    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season, "scan_message": scan_message},
    )


@router.post("/scene-jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_scene_job(
    request: Request, job_id: int, title_id: int | None = Form(None), detail_view: bool = Form(False)
):
    """Reachable from three places: the standalone queue page (no title_id --
    re-renders the queue table), the standalone scene-review page (title_id set,
    detail_view False -- re-renders scene_review_list.html), and the title detail
    card (title_id set, detail_view True -- re-renders the whole card)."""
    success, error = await scene_job_queue.cancel_job(job_id)
    if title_id is not None:
        return await _render_scene_review(request, title_id, detail_view)
    from app.routers.queue import _render_queue_partial

    return await _render_queue_partial(request, cancel_error=None if success else error)


@router.post("/scenes/{scene_id}/approve", response_class=HTMLResponse)
async def approve_scene(
    request: Request, scene_id: int, title_id: int | None = Form(None), detail_view: bool = Form(False)
):
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        # Fall back to the title_id the form itself was rendered for (a rescan
        # can delete pending/rejected scenes out from under an open review
        # page -- see scan_for_scenes) rather than going blank with no
        # indication the click did nothing; still re-renders the (now fresh)
        # review list for that title even though this specific scene is gone.
        resolved_title_id = scene.title_id if scene is not None else title_id
        if scene is not None:
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = datetime.datetime.utcnow()
            await session.commit()
    if resolved_title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, resolved_title_id, detail_view)


@router.post("/scenes/{scene_id}/reject", response_class=HTMLResponse)
async def reject_scene(
    request: Request, scene_id: int, title_id: int | None = Form(None), detail_view: bool = Form(False)
):
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        resolved_title_id = scene.title_id if scene is not None else title_id
        if scene is not None:
            scene.status = SceneReviewStatus.rejected
            scene.reviewed_at = datetime.datetime.utcnow()
            await session.commit()
    if resolved_title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, resolved_title_id, detail_view)


@router.post("/scenes/{scene_id}/toggle-mute", response_class=HTMLResponse)
async def toggle_scene_mute(
    request: Request, scene_id: int, title_id: int | None = Form(None), detail_view: bool = Form(False)
):
    """Create an explicit audio override by flipping the effective audio rule."""
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        resolved_title_id = scene.title_id if scene is not None else title_id
        if scene is not None:
            policy = await get_setting(session, "scene_content_policy")
            scene.audio_override = not scene_decision(scene, policy).mute_audio
            scene.mute_audio = scene.audio_override
            await session.commit()
    if resolved_title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, resolved_title_id, detail_view)


@router.post("/library/title/{title_id}/reset-scene-audio", response_class=HTMLResponse)
async def reset_scene_audio(request: Request, title_id: int, detail_view: bool = Form(False)):
    """Explicitly discard legacy/per-scene audio choices; never regenerate here."""
    async with get_session() as session:
        if not await _claim_review_generation(session, request, title_id):
            return await _render_scene_review(request, title_id, detail_view)
        result = await session.execute(select(DetectedScene).where(DetectedScene.title_id == title_id))
        for scene in result.scalars():
            scene.audio_override = None
            scene.mute_audio = False
        await session.commit()
    return await _render_scene_review(request, title_id, detail_view)


@router.post("/scenes/{scene_id}/action", response_class=HTMLResponse)
async def set_scene_action(request: Request, scene_id: int, action: str = Form(...),
                           title_id: int | None = Form(None), detail_view: bool = Form(False)):
    """An explicit manual content action takes precedence over category policy."""
    if action not in {"inherit", *(a.value for a in Action)}:
        raise HTTPException(422, "Invalid scene action")
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        resolved_title_id = scene.title_id if scene else title_id
        if scene is not None:
            scene.action_override = None if action == "inherit" else action
            scene.audio_override = None if action == "inherit" else action == "blur_mute"
            scene.mute_audio = bool(scene.audio_override)
            await session.commit()
    return await _render_scene_review(request, resolved_title_id, detail_view) if resolved_title_id else HTMLResponse("")


@router.post("/library/title/{title_id}/approve-high-confidence", response_class=HTMLResponse)
async def approve_high_confidence_scenes(request: Request, title_id: int, detail_view: bool = Form(False)):
    """Bulk-approves every still-pending candidate whose dense verification pass
    (see app.scenes.pipeline.scan_for_scenes) cleared scene_high_confidence_fraction
    -- the review-friction reducer for the clear-cut cases, while anything short/
    borderline enough to score low on that fraction still needs an actual look."""
    async with get_session() as session:
        if not await _claim_review_generation(session, request, title_id):
            return await _render_scene_review(request, title_id, detail_view)
        high_confidence_fraction = float(await get_setting(session, "scene_high_confidence_fraction"))
        result = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.pending,
                DetectedScene.verified_fraction.is_not(None),
                DetectedScene.verified_fraction >= high_confidence_fraction,
            )
        )
        now = datetime.datetime.utcnow()
        for scene in result.scalars().all():
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = now
        await session.commit()
    return await _render_scene_review(request, title_id, detail_view)


@router.post("/library/title/{title_id}/approve-all-scenes", response_class=HTMLResponse)
async def approve_all_scenes(request: Request, title_id: int, detail_view: bool = Form(False)):
    """Bulk-approves every still-pending candidate regardless of verified_fraction
    -- the blunt version of approve-high-confidence above, for anyone who'd rather
    just approve everything a scan found than review one at a time."""
    async with get_session() as session:
        if not await _claim_review_generation(session, request, title_id):
            return await _render_scene_review(request, title_id, detail_view)
        result = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.pending,
            )
        )
        now = datetime.datetime.utcnow()
        for scene in result.scalars().all():
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = now
        await session.commit()
    return await _render_scene_review(request, title_id, detail_view)


# Padding on each side of the requested [start, end], for the same reason the
# preview player pads it -- lets the reviewer judge the boundaries, not just
# confirm the flagged moment.
_CLIP_PAD_SECONDS = 3.0
_CLIP_MAX_DURATION_SECONDS = 60.0  # sanity cap regardless of what a caller requests


def _extract_clip_blocking(ffmpeg_bin: str, video_path: Path, start: float, duration: float, out_path: Path) -> None:
    proc = subprocess.run(
        [
            ffmpeg_bin, "-nostdin", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
            # Re-encode to H.264/AAC regardless of the source codec -- real-world
            # rips are often HEVC video and/or AC3/DTS audio, neither of which any
            # mainstream browser decodes natively, MP4 container or not. A clip
            # this short (seconds, capped well below a minute) re-encodes in
            # roughly real time or faster even on modest hardware, so the cost is
            # negligible against the guarantee of it actually playing.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg clip extraction failed at {start:.2f}s+{duration:.2f}s in {video_path}")


@router.get("/library/title/{title_id}/scene-clip")
async def scene_clip(title_id: int, start: float, end: float):
    """A short, re-encoded (guaranteed browser-playable) MP4 clip covering
    [start, end] plus padding -- replaces an earlier whole-file Range-streaming
    approach that turned out not to work in practice: most real media here is
    .mkv (inconsistent <video> container support across browsers even when the
    codecs would otherwise play) and/or HEVC/AC3 (which browsers can't decode
    natively at all, container aside). A short clip sidesteps both problems at
    once instead of trying to solve either."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None:
        raise HTTPException(status_code=404)

    video_path = Path(title.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404)

    clip_start = max(0.0, start - _CLIP_PAD_SECONDS)
    clip_duration = min(_CLIP_MAX_DURATION_SECONDS, (end - start) + 2 * _CLIP_PAD_SECONDS)

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_path = Path(tmpdir) / "clip.mp4"
        try:
            await asyncio.to_thread(
                _extract_clip_blocking, app_settings.ffmpeg_bin, video_path, clip_start, clip_duration, clip_path
            )
        except RuntimeError:
            raise HTTPException(status_code=404)
        data = clip_path.read_bytes()

    return Response(content=data, media_type="video/mp4")


@router.get("/library/title/{title_id}/scene-thumbnail")
async def scene_thumbnail(title_id: int, t: float):
    """One representative frame at timestamp t, for the review list's thumbnail
    and the preview player's poster image -- so approving isn't just trusting the
    classifier's confidence number blind.

    Cached aggressively (title_id+t is a deterministic key -- the same source
    frame every time, since the original file is never modified in place) so
    the review list's 3-second self-poll while a scan/blur job is running
    doesn't re-run ffmpeg and re-fetch every thumbnail over the network on each
    tick -- that was visibly flashing every image on the page each poll. The
    one edge case this doesn't cover is Sonarr/Radarr silently replacing the
    underlying file with a different cut -- same staleness caveat that already
    applies to the scene timestamps themselves, not new here."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None:
        raise HTTPException(status_code=404)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "thumb.jpg"
        try:
            await asyncio.to_thread(
                extract_frame, app_settings.ffmpeg_bin, Path(title.video_path), max(0.0, t), frame_path
            )
        except RuntimeError:
            raise HTTPException(status_code=404)
        data = frame_path.read_bytes()

    return Response(
        content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )


@router.post("/scenes/{scene_id}/adjust", response_class=HTMLResponse)
async def adjust_scene(
    request: Request,
    scene_id: int,
    start_seconds: float = Form(...),
    end_seconds: float = Form(...),
    title_id: int | None = Form(None),
    detail_view: bool = Form(False),
):
    from app.mux.remux import probe

    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        # Fall back to the title_id the form itself was rendered for -- see
        # approve_scene's comment for why (a concurrent rescan can delete this
        # scene out from under an open review page).
        resolved_title_id = scene.title_id if scene is not None else title_id
        if scene is not None and end_seconds > start_seconds:
            title = await session.get(Title, scene.title_id)
            duration = None
            if title is not None:
                try:
                    probe_data = await probe(app_settings.ffprobe_bin, Path(title.video_path))
                    duration = float(probe_data["format"].get("duration", 0)) or None
                except Exception:
                    # Best-effort clamp only -- if the probe fails (file moved,
                    # ffprobe error), fall back to trusting the submitted value
                    # rather than blocking the edit entirely.
                    duration = None
            scene.start_seconds = max(0.0, start_seconds)
            scene.end_seconds = min(end_seconds, duration) if duration else end_seconds
            await session.commit()
    if resolved_title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, resolved_title_id, detail_view)


# ---------------------------------------------------------------------------
# Manual scenes: user-marked windows from the scene review page's scrub timeline
# ---------------------------------------------------------------------------

_build_tasks: set[asyncio.Task] = set()  # strong refs so fire-and-forget builds aren't GC'd mid-run


async def _title_or_404(title_id: int) -> Title:
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None or not Path(title.video_path).exists():
        raise HTTPException(status_code=404)
    return title


@router.get("/library/title/{title_id}/scrub/status")
async def scrub_status(title_id: int):
    title = await _title_or_404(title_id)
    return JSONResponse(scrub_proxy.status_for(title))


@router.post("/library/title/{title_id}/scrub/build")
async def scrub_build(title_id: int):
    title = await _title_or_404(title_id)
    if not scrub_proxy.is_ready(title):
        task = asyncio.create_task(scrub_proxy.build(title))
        _build_tasks.add(task)
        task.add_done_callback(_build_tasks.discard)
        # Let the build register itself so the immediate status poll sees "building".
        await asyncio.sleep(0)
    return JSONResponse(scrub_proxy.status_for(title))


@router.get("/library/title/{title_id}/scrub/sprite.jpg")
async def scrub_sprite(title_id: int):
    title = await _title_or_404(title_id)
    path = scrub_proxy.sprite_path_for(title)
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/library/title/{title_id}/scrub/video.mp4")
async def scrub_video(title_id: int, request: Request):
    """Range-capable (browsers need it to seek) -- the Starlette version pinned
    here predates FileResponse's own Range support, so this does it by hand."""
    title = await _title_or_404(title_id)
    path = scrub_proxy.video_path_for(title)
    if not path.exists():
        raise HTTPException(status_code=404)
    size = path.stat().st_size
    start, end = 0, size - 1
    status = 200
    range_header = request.headers.get("range", "")
    if range_header.startswith("bytes="):
        first, _, last = range_header[6:].split(",")[0].partition("-")
        try:
            if first == "":  # suffix range: last N bytes
                start = max(0, size - int(last))
            else:
                start = int(first)
                if last:
                    end = min(int(last), size - 1)
        except ValueError:
            raise HTTPException(status_code=416)
        if start > end or start >= size:
            raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206

    def chunks():
        remaining = end - start + 1
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(1024 * 256, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=status, media_type="video/mp4", headers=headers)


@router.post("/library/title/{title_id}/scenes/manual", response_class=HTMLResponse)
async def add_manual_scenes(
    request: Request,
    title_id: int,
    starts: list[float] = Form(default=[]),
    ends: list[float] = Form(default=[]),
    mutes: list[str] = Form(default=[]),
    detail_view: bool = Form(False),
):
    """Creates one already-approved scene per (start, end) pair. Approved from the
    start because the user marked these by hand -- there's nothing left to review,
    and a fresh rescan replaces these windows unless Keep manual scenes is selected."""
    from app.mux.remux import probe

    duration = None
    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is None:
            raise HTTPException(status_code=404)
        try:
            probe_data = await probe(app_settings.ffprobe_bin, Path(title.video_path))
            duration = float(probe_data["format"].get("duration", 0)) or None
        except Exception:
            # Best-effort clamp only, same as adjust_scene.
            duration = None

        now = datetime.datetime.utcnow()
        # "1" explicitly forces mute; unchecked/missing inherits global audio.
        mute_flags = [m == "1" for m in mutes] + [False] * len(starts)
        for start, end, mute in zip(starts, ends, mute_flags):
            start = max(0.0, start)
            if duration:
                end = min(end, duration)
            if end <= start:
                continue
            session.add(
                DetectedScene(
                    id=secrets.randbits(62) + 1,
                    title_id=title_id,
                    scan_generation=title.scene_scan_generation,
                    start_seconds=start,
                    end_seconds=end,
                    peak_confidence=1.0,
                    mute_audio=mute,
                    audio_override=True if mute else None,
                    source="manual",
                    status=SceneReviewStatus.approved,
                    reviewed_at=now,
                )
            )
        await session.commit()
    return await _render_scene_review(request, title_id, detail_view)
