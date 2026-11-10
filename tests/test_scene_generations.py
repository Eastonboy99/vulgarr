"""Transactional generation tests against real SQLite, with no media mutation."""
import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select, update, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import Base, DetectedScene, Title, SceneJob, JobState, MediaType, SceneJobKind
from app.domain import SceneReviewStatus
from app.scenes.pipeline import _activate_scan
from app.queue import scene_worker


def run(coro):
    return asyncio.run(coro)


def candidate(title_id=1, **updates):
    return DetectedScene(title_id=title_id, start_seconds=1, end_seconds=2,
                         peak_confidence=.9, status=SceneReviewStatus.approved, **updates)


async def database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Title(id=1, media_type=MediaType.movie, display_name="Test",
                          video_path="/original.mkv", vulgarr_edit_path="/last-good.mkv"))
        await session.commit()
    return engine, factory


def test_three_fresh_generations_replace_approved_and_manual(tmp_path):
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            async with factory() as session:
                session.add_all([candidate(source="detected"), candidate(source="manual")])
                await session.commit()
                for number in range(3):
                    job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.processing)
                    session.add(job)
                    await session.flush()
                    await session.execute(update(Title).where(Title.id == 1).values(scene_scan_generation=job.id))
                    await session.commit()
                    await _activate_scan(session, 1, job.id, [candidate(scan_generation=job.id)])
                    rows = (await session.execute(select(DetectedScene))).scalars().all()
                    assert len(rows) == 1
                    assert rows[0].scan_generation == job.id
                    assert (await session.get(Title, 1)).vulgarr_edit_path == "/last-good.mkv"
                    # Successful retry cannot wipe subsequent review decisions.
                    rows[0].status = SceneReviewStatus.rejected
                    await session.commit()
                    with pytest.raises(ValueError, match="completed"):
                        await _activate_scan(session, 1, job.id, [candidate()])
                    rows = (await session.execute(select(DetectedScene))).scalars().all()
                    assert rows[0].status == SceneReviewStatus.rejected
        finally:
            await engine.dispose()
    run(scenario())


@pytest.mark.parametrize("keep_manual", [False, True])
def test_explicit_manual_preservation(tmp_path, keep_manual):
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            async with factory() as session:
                session.add_all([candidate(source="manual"), candidate()])
                job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.processing, keep_manual=keep_manual)
                session.add(job)
                await session.flush()
                await session.execute(update(Title).values(scene_scan_generation=job.id))
                await session.commit()
                await _activate_scan(session, 1, job.id, [candidate(scan_generation=job.id)])
                rows = (await session.execute(select(DetectedScene))).scalars().all()
                assert len(rows) == (2 if keep_manual else 1)
                assert sum(s.source == "manual" for s in rows) == int(keep_manual)
        finally:
            await engine.dispose()
    run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_late_or_cancelled_generation_cannot_activate(tmp_path, cancelled):
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            async with factory() as session:
                session.add(candidate())
                job = SceneJob(title_id=1, kind=SceneJobKind.scan,
                               state=JobState.cancelled if cancelled else JobState.processing)
                session.add(job)
                await session.flush()
                job_id = job.id
                await session.execute(update(Title).values(scene_scan_generation=job_id if cancelled else job_id + 1))
                await session.commit()
                with pytest.raises(ValueError):
                    await _activate_scan(session, 1, job_id, [])
                assert len((await session.execute(select(DetectedScene))).scalars().all()) == 1
                assert (await session.get(Title, 1)).vulgarr_edit_path == "/last-good.mkv"
        finally:
            await engine.dispose()
    run(scenario())


def test_concurrent_single_bulk_claims_are_database_backed(tmp_path, monkeypatch):
    async def scenario():
        engine, factory = await database(tmp_path)
        @asynccontextmanager
        async def sessions():
            async with factory() as session:
                yield session
        monkeypatch.setattr(scene_worker, "get_session", sessions)
        try:
            one, two = scene_worker.SceneJobQueue(), scene_worker.SceneJobQueue()
            results = await asyncio.gather(one.enqueue_scan(1), two.enqueue_scan_for_titles_if_not_already_active([1, 1]))
            assert sum(bool(r) for r in results) == 1
            async with factory() as session:
                jobs = (await session.execute(select(SceneJob))).scalars().all()
                assert len(jobs) == 1
                assert (await session.get(Title, 1)).scene_scan_generation == jobs[0].id
            assert await one.enqueue_blur(1) == 0
        finally:
            await engine.dispose()
    run(scenario())


def test_semantic_scan_proposes_bikini_without_nudenet_and_keeps_allowed_evidence(tmp_path, monkeypatch):
    from app.scenes import pipeline
    from app.scenes.content_policy import VISUAL_CATEGORIES, scene_decision, DEFAULT_POLICY, Action
    from app.db.session import set_setting
    from types import SimpleNamespace
    async def scenario():
        engine, factory = await database(tmp_path)
        async def probe(*args):
            return {"format": {"duration": "20"}}
        async def semantic(**options):
            return [(0, 10, {"schema_version": 1, "coverage": sorted(VISUAL_CATEGORIES),
                "labels": {"clothing.bikini": .95}, "uncertain": False,
                "explanation": "Mock bikini", "model": "mock-v1"}),
                (10, 20, {"schema_version": 1, "coverage": sorted(VISUAL_CATEGORIES),
                "labels": {"clothing.low_cut": .95}, "uncertain": False,
                "explanation": "Mock low neckline", "model": "mock-v1"})]
        async def forbidden(*args, **options):
            raise AssertionError("Semantic coverage must not depend on NudeNet candidates")
        monkeypatch.setattr(pipeline, "probe", probe)
        monkeypatch.setattr(pipeline, "scan_content_windows", semantic)
        monkeypatch.setattr(pipeline, "scan_video_frames", forbidden)
        monkeypatch.setattr(pipeline, "app_settings", SimpleNamespace(data_dir=tmp_path, ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg"))
        try:
            async with factory() as session:
                await set_setting(session, "scene_semantic_enabled", True)
                job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.processing)
                session.add(job)
                await session.flush()
                title = await session.get(Title, 1)
                title.scene_scan_generation = job.id
                await session.commit()
                outcome = await pipeline.scan_for_scenes(session, title, job.id)
                assert outcome.candidate_count == 2
                rows = (await session.execute(select(DetectedScene).order_by(DetectedScene.start_seconds))).scalars().all()
                assert rows[0].status == rows[1].status == SceneReviewStatus.approved
                assert scene_decision(rows[0], DEFAULT_POLICY).action == Action.blur_mute
                assert scene_decision(rows[1], DEFAULT_POLICY).action == Action.allow
                assert title.scene_analysis is not None
        finally:
            await engine.dispose()
    run(scenario())


def test_stale_apply_is_fenced_before_publication(tmp_path, monkeypatch):
    from app.scenes import pipeline
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            async with factory() as session:
                scene = candidate()
                session.add(scene)
                await session.commit()
                scene_id = scene.id
                title = await session.get(Title, 1)
                async def renderer(**options):
                    async with factory() as reviewer:
                        changed = await reviewer.get(DetectedScene, scene_id)
                        changed.status = SceneReviewStatus.rejected
                        await reviewer.commit()
                    await options["before_publish"]()
                    raise AssertionError("Obsolete rendering must not publish")
                monkeypatch.setattr(pipeline, "apply_blur", renderer)
                with pytest.raises(ValueError, match="changed during Apply"):
                    await pipeline.apply_scene_blur(session, title, "test")
                await session.rollback()
                assert (await session.get(Title, 1)).vulgarr_edit_path == "/last-good.mkv"
        finally:
            await engine.dispose()
    run(scenario())


def test_local_scan_uses_selected_covered_label_and_persists_real_evidence(tmp_path, monkeypatch):
    import copy
    import json
    from types import SimpleNamespace
    from app.scenes import pipeline
    from app.scenes.content_policy import DEFAULT_POLICY, scene_decision, Action
    from app.vision import classifier
    from app.vision.scene_cluster import FrameScore
    from app.db.session import set_setting
    async def scenario():
        engine, factory = await database(tmp_path)
        detections = [{"class": "FEMALE_BREAST_COVERED", "score": .9}]
        async def probe(*args):
            return {"format": {"duration": "20"}}
        async def scan(*args, **options):
            return [FrameScore(t, classifier.frame_confidence(detections), detections) for t in (5, 5.5)]
        async def refine(*args):
            return (4.5, 6, 1.0, [FrameScore(5, .9, detections)])
        async def forbidden(*args, **options):
            raise AssertionError("Local label policy must not call a swimwear-allowing semantic verifier")
        monkeypatch.setattr(pipeline, "probe", probe)
        monkeypatch.setattr(pipeline, "scan_video_frames", scan)
        monkeypatch.setattr(pipeline, "_refine_candidate_boundary", refine)
        monkeypatch.setattr(pipeline, "verify_candidate", forbidden)
        monkeypatch.setattr(pipeline, "app_settings", SimpleNamespace(data_dir=tmp_path, ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg"))
        try:
            async with factory() as session:
                policy = copy.deepcopy(DEFAULT_POLICY)
                policy["actions"]["nudenet.FEMALE_BREAST_COVERED"] = "blur_mute"
                await set_setting(session, "scene_content_policy", policy)
                await set_setting(session, "claude_vision_verify_enabled", True)
                job = SceneJob(title_id=1, kind=SceneJobKind.scan, state=JobState.processing)
                session.add(job)
                await session.flush()
                title = await session.get(Title, 1)
                title.scene_scan_generation = job.id
                await session.commit()
                outcome = await pipeline.scan_for_scenes(session, title, job.id)
                assert outcome.candidate_count == outcome.auto_approved_count == 1
                row = (await session.execute(select(DetectedScene))).scalar_one()
                assert json.loads(row.category_evidence)["labels"] == {"nudenet.FEMALE_BREAST_COVERED": .9}
                assert scene_decision(row, policy).action == Action.blur_mute
                policy["actions"]["nudenet.FEMALE_BREAST_COVERED"] = "allow"
                assert scene_decision(row, policy).action == Action.allow
                coverage = json.loads(title.scene_analysis)["coverage"]
                assert "nudenet.FEMALE_BREAST_COVERED" in coverage
                assert "nudenet.FACE_FEMALE" not in coverage
        finally:
            await engine.dispose()
    run(scenario())


def test_bulk_review_rejects_obsolete_generation(tmp_path):
    from app.routers.scenes import _claim_review_generation
    from types import SimpleNamespace
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            async with factory() as session:
                await session.execute(update(Title).values(scene_scan_generation=42, scene_scan_status="scanned"))
                await session.commit()
                async def old_form():
                    return {"scan_generation": "41"}
                async def new_form():
                    return {"scan_generation": "42"}
                assert not await _claim_review_generation(session, SimpleNamespace(form=old_form), 1)
                assert await _claim_review_generation(session, SimpleNamespace(form=new_form), 1)
                await session.commit()
                await session.execute(update(Title).values(scene_scan_status="scanning"))
                await session.commit()
                assert not await _claim_review_generation(session, SimpleNamespace(form=new_form), 1)
        finally:
            await engine.dispose()
    run(scenario())


def test_legacy_audio_migration_runs_once_and_preserves_both_values(tmp_path):
    from app.db.session import _run_column_migrations
    async def scenario():
        engine, factory = await database(tmp_path)
        try:
            # Recreate only the scene table in its pre-policy form.
            async with engine.begin() as conn:
                await conn.execute(text("DROP TABLE detected_scenes"))
                await conn.execute(text("CREATE TABLE detected_scenes (id INTEGER PRIMARY KEY, mute_audio BOOLEAN NOT NULL)"))
                await conn.execute(text("INSERT INTO detected_scenes VALUES (1, 0), (2, 1)"))
                await _run_column_migrations(conn)
                rows = (await conn.execute(text("SELECT audio_override FROM detected_scenes ORDER BY id"))).all()
                assert rows == [(0,), (1,)]
                await conn.execute(text("UPDATE detected_scenes SET audio_override = NULL"))
                await _run_column_migrations(conn)
                assert (await conn.execute(text("SELECT audio_override FROM detected_scenes"))).all() == [(None,), (None,)]
        finally:
            await engine.dispose()
    run(scenario())
