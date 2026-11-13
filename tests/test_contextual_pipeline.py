"""Hybrid scan atomicity and review-only contracts; no real inference/media edits."""
import asyncio
import json
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import Base, Title, DetectedScene, SceneJob, JobState, SceneJobKind, MediaType
from app.db.session import DEFAULT_SETTINGS, set_setting
from app.domain import SceneReviewStatus
from app.scenes import pipeline
from app.scenes.content_policy import DEFAULT_POLICY, scene_decision, Action
from app.vision import contextual


def evidence(state="contextual_nudity"):
    return contextual.observations_to_evidence({"frames": [{"frame_id": f"f{i}", "state": state, "evidence": "Synthetic observation"} for i in range(4)], "window_summary": "Synthetic summary"}, "mock")


async def database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'hybrid.db'}")
    async with engine.begin() as connection: await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Title(id=1, media_type=MediaType.movie, display_name="Fixture", video_path="/original.mkv", vulgarr_edit_path="/last-good.mkv"))
        session.add(DetectedScene(id=123, title_id=1, start_seconds=1, end_seconds=2, peak_confidence=.9, status=SceneReviewStatus.approved, source="manual"))
        await session.commit()
    return engine, factory


@pytest.mark.parametrize("mode", ["success", "failure", "changed_settings", "changed_labels", "cancelled"])
def test_independent_discovery_with_no_nudenet_hits_and_atomic_activation(tmp_path, monkeypatch, mode):
    async def scenario():
        engine, factory = await database(tmp_path)
        async def probe(*args): return {"format": {"duration": "24"}}
        async def nudenet(*args, **kwargs): return []
        current_session = []
        async def local(*args):
            if mode == "changed_labels":
                policy = json.loads(json.dumps(DEFAULT_POLICY))
                policy["actions"]["nudenet.BELLY_EXPOSED"] = "blur"
                await set_setting(current_session[0], "scene_content_policy", policy)
            return None
        async def forbidden(*args, **kwargs): raise AssertionError("NudeNet cannot refine or veto contextual discoveries")
        monkeypatch.setattr(pipeline, "probe", probe)
        monkeypatch.setattr(pipeline, "scan_video_frames", nudenet)
        monkeypatch.setattr(pipeline, "_refine_candidate_boundary", forbidden)
        monkeypatch.setattr(pipeline, "verify_candidate", forbidden)
        monkeypatch.setattr(contextual, "local_endpoint", local)
        monkeypatch.setattr(pipeline, "app_settings", SimpleNamespace(data_dir=tmp_path, ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg"))
        try:
            async with factory() as session:
                current_session.append(session)
                for key, value in {"scene_contextual_enabled": True, "scene_contextual_base_url": "http://private/v1", "scene_contextual_model": "mock", "scene_auto_process": True}.items():
                    await set_setting(session, key, value)
                job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.processing)
                session.add(job)
                await session.flush()
                title = await session.get(Title, 1)
                title.scene_scan_generation = job.id
                await session.commit()
                async def scan(**options):
                    assert options["duration"] == 24  # Whole title, not NudeNet windows.
                    if mode == "failure": raise ValueError("outage")
                    if mode == "changed_settings": await set_setting(session, "scene_contextual_model", "changed")
                    if mode == "cancelled":
                        job.state = JobState.cancelled
                        await session.commit()
                    return [(0, 8, evidence("clothed")), (8, 16, evidence()), (16, 24, evidence())]
                monkeypatch.setattr(contextual, "scan_contextual_windows", scan)
                if mode == "success":
                    outcome = await pipeline.scan_for_scenes(session, title, job.id)
                    assert outcome.candidate_count == outcome.need_review_count == 1
                    assert outcome.auto_approved_count == 0
                    rows = (await session.execute(select(DetectedScene))).scalars().all()
                    assert len(rows) == 1 and rows[0].id != 123
                    assert rows[0].start_seconds == 8 and rows[0].end_seconds == 24
                    assert rows[0].status == SceneReviewStatus.pending
                    assert rows[0].verified_fraction is None
                    decision = scene_decision(rows[0], DEFAULT_POLICY)
                    assert decision.action == Action.blur_mute and decision.needs_review
                    assert "nudity.contextual" in json.loads(title.scene_analysis)["coverage"]
                    assert json.loads(title.scene_analysis)["contextual"]["windows"] == 3
                else:
                    with pytest.raises(ValueError): await pipeline.scan_for_scenes(session, title, job.id)
                    rows = (await session.execute(select(DetectedScene))).scalars().all()
                    assert [r.id for r in rows] == [123]
                assert (await session.get(Title, 1)).vulgarr_edit_path == "/last-good.mkv"
        finally: await engine.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize("pending", [0, 1])
def test_auto_apply_waits_for_review(tmp_path, monkeypatch, pending):
    from app.queue import scene_worker
    async def scenario():
        engine, factory = await database(tmp_path)
        @asynccontextmanager
        async def sessions():
            async with factory() as session:
                yield session
        async def scan(*args, **kwargs):
            return pipeline.ScanOutcome(1, 0 if pending else 1, pending)
        queued = []
        async def enqueue(title_id): queued.append(title_id)
        monkeypatch.setattr(scene_worker, "get_session", sessions)
        monkeypatch.setattr(scene_worker, "scan_for_scenes", scan)
        monkeypatch.setattr(scene_worker, "remote_extract_enabled", lambda: True)
        queue = scene_worker.SceneJobQueue()
        monkeypatch.setattr(queue, "enqueue_blur_if_not_already_active", enqueue)
        try:
            async with factory() as session:
                await set_setting(session, "scene_auto_process", True)
                job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.queued)
                session.add(job)
                await session.commit()
                job_id = job.id
            await queue._run_job(job_id)
            assert queued == ([] if pending else [1])
            async with factory() as session:
                assert (await session.get(Title, 1)).vulgarr_edit_path == "/last-good.mkv"
        finally: await engine.dispose()
    asyncio.run(scenario())


def test_legacy_verifier_cannot_veto_contextual_rows(tmp_path, monkeypatch):
    async def scenario():
        engine, factory = await database(tmp_path)
        async def forbidden(*args, **kwargs): raise AssertionError("Legacy verifier must not reinterpret contextual evidence")
        monkeypatch.setattr(pipeline, "verify_candidate", forbidden)
        monkeypatch.setattr(pipeline, "classify_window", forbidden)
        try:
            async with factory() as session:
                row = await session.get(DetectedScene, 123)
                row.category_evidence = json.dumps(evidence())
                row.status = SceneReviewStatus.pending
                await set_setting(session, "claude_vision_verify_enabled", True)
                await session.commit()
                await pipeline.reverify_scenes_with_claude(session, await session.get(Title, 1))
                assert row.status == SceneReviewStatus.pending
                assert json.loads(row.category_evidence)["labels"] == {"nudity.contextual": 1.0}
        finally: await engine.dispose()
    asyncio.run(scenario())


def test_overlapping_detector_coverage_subtracted_without_losing_missing_segments(tmp_path, monkeypatch):
    async def get_setting(session, key): return DEFAULT_SETTINGS[key]
    async def scan(**options):
        return [(0, 8, evidence()), (8, 16, evidence()), (16, 24, evidence("uncertain"))]
    monkeypatch.setattr(pipeline, "get_setting", get_setting)
    from app.scenes import protection
    monkeypatch.setattr(protection, "get_setting", get_setting)
    monkeypatch.setattr(contextual, "scan_contextual_windows", scan)
    config = {key: DEFAULT_SETTINGS[key] for key in contextual.KEYS}
    config.update(scene_contextual_enabled=True, scene_contextual_model="mock")
    original = DetectedScene(id=123, title_id=1, start_seconds=6, end_seconds=10, peak_confidence=.9, status=SceneReviewStatus.approved)
    rows = [original]
    result = asyncio.run(pipeline._add_contextual_rows(None, SimpleNamespace(id=1, video_path="/original"), 1, rows,
        tmp_path, "source", 24, config, None, None))
    assert result["added_rows"] == 3
    assert rows[0] is original and original.status == SceneReviewStatus.approved
    assert [(r.start_seconds, r.end_seconds) for r in rows[1:]] == [(0, 6), (10, 16), (16, 24)]
    assert all(r.status == SceneReviewStatus.pending for r in rows[1:])
    assert not json.loads(rows[-1].category_evidence)["labels"]
    assert json.loads(rows[-1].category_evidence)["uncertain"]
