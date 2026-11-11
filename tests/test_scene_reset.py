"""Explicit clear-now/restart behavior, distinct from atomic normal rescans."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import (
    Base, DetectedScene, Title, SceneJob, ProcessingJob, JobState, MediaType,
    SceneJobKind, SceneReviewStatus, TriggerSource,
)
from app.queue import scene_worker
from app.routers import scenes, library
from app.scenes.pipeline import _activate_scan


@asynccontextmanager
async def database(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'reset.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    @asynccontextmanager
    async def sessions():
        async with factory() as session:
            yield session
    queue = scene_worker.SceneJobQueue()
    for module in (scene_worker, scenes, library):
        monkeypatch.setattr(module, 'get_session', sessions)
    monkeypatch.setattr(scenes, 'scene_job_queue', queue)
    try:
        yield factory, queue
    finally:
        await engine.dispose()


def title(number, *, series=None, season=1):
    return Title(id=number, display_name=f'Title {number}', video_path=f'/original-{number}.mkv',
        media_type=MediaType.movie if series is None else MediaType.episode,
        sonarr_series_id=series, series_title=f'Show {series}' if series is not None else None,
        season_number=season if series is not None else None,
        episode_number=number if series is not None else None,
        vulgarr_edit_path=f'/last-good-{number}.mkv', clean_audio_path=f'/clean-{number}.mka',
        matched_cue_count=17, scene_scan_status='scanned')


def window(number, source='detected', status=SceneReviewStatus.approved):
    return DetectedScene(title_id=number, start_seconds=1, end_seconds=2,
                         peak_confidence=.9, source=source, status=status)


def test_reset_clears_all_review_states_and_manual_windows_without_touching_media(tmp_path, monkeypatch):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            original, published = tmp_path/'source.mkv', tmp_path/'edited.mkv'
            original.write_bytes(b'original'); published.write_bytes(b'last successful edit')
            async with factory() as session:
                movie = title(1)
                movie.video_path = str(original)
                movie.vulgarr_edit_path = str(published)
                session.add(movie)
                await session.flush()
                session.add_all([window(1, status=status) for status in SceneReviewStatus])
                session.add(window(1, source='manual'))
                await session.commit()
            result = await queue.clear_and_restart_scans([1, 1])
            assert result.restarted == 1 and result.cleared == 4 and result.busy == 0
            async with factory() as session:
                assert (await session.execute(select(DetectedScene))).scalars().all() == []
                movie = await session.get(Title, 1)
                jobs = (await session.execute(select(SceneJob))).scalars().all()
                assert len(jobs) == 1 and jobs[0].state == JobState.queued
                assert movie.scene_scan_generation == jobs[0].id
                assert movie.scene_scan_status == 'queued' and movie.scene_analysis is None
                assert movie.clean_audio_path == '/clean-1.mka' and movie.matched_cue_count == 17
                assert movie.vulgarr_edit_path == str(published)
                # A failed replacement does not resurrect intentionally cleared rows.
                jobs[0].state = JobState.failed
                await session.commit()
                assert (await session.execute(select(DetectedScene))).scalars().all() == []
            assert original.read_bytes() == b'original'
            assert published.read_bytes() == b'last successful edit'
    asyncio.run(check())


@pytest.mark.parametrize('state', [JobState.queued, JobState.processing])
def test_reset_supersedes_scan_and_prevents_late_activation(tmp_path, monkeypatch, state):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                movie = title(1); session.add(movie)
                old = SceneJob(title_id=1, kind=SceneJobKind.scan, state=state)
                session.add(old); await session.flush()
                movie.scene_scan_generation = old.id
                session.add(window(1)); await session.commit()
                old_id = old.id
            cancelled = []
            queue._running[old_id] = SimpleNamespace(cancel=lambda: cancelled.append(old_id))
            old_cache = tmp_path/'scan_work'/str(old_id)/'scores.jsonl'
            old_cache.parent.mkdir(parents=True); old_cache.write_text('old partial inference')
            result = await queue.clear_and_restart_scans([1])
            assert result.restarted == 1 and cancelled == [old_id]
            async with factory() as session:
                old = await session.get(SceneJob, old_id)
                movie = await session.get(Title, 1)
                assert old.state == JobState.cancelled
                assert movie.scene_scan_generation != old_id
                assert not (tmp_path/'scan_work'/str(movie.scene_scan_generation)).exists()
                assert old_cache.read_text() == 'old partial inference'
                with pytest.raises(ValueError, match='superseded'):
                    await _activate_scan(session, 1, old_id, [window(1)])
                assert (await session.execute(select(DetectedScene))).scalars().all() == []
            # Even an old ID still present in the delivery queue cannot run.
            await queue._run_job(old_id)
    asyncio.run(check())


def test_remote_process_reset_keeps_old_worker_cancelled_and_new_generation_queued(tmp_path, monkeypatch):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add(title(1)); session.add(window(1)); await session.commit()
            old_id = await queue.enqueue_scan(1)
            async def late_scan(session, movie, job_id, **options):
                # No local task registration: simulate another process issuing
                # the reset, so the durable fence—not Task.cancel—must stop it.
                await queue.clear_and_restart_scans([1])
                await _activate_scan(session, movie.id, job_id, [window(1)])
            monkeypatch.setattr(scene_worker, 'scan_for_scenes', late_scan)
            await queue._run_job(old_id)
            async with factory() as session:
                assert (await session.get(SceneJob, old_id)).state == JobState.cancelled
                movie = await session.get(Title, 1)
                assert movie.scene_scan_generation != old_id and movie.scene_scan_status == 'queued'
                assert (await session.execute(select(DetectedScene))).scalars().all() == []
    asyncio.run(check())


@pytest.mark.parametrize('kind', ['audio', 'blur', 'claude_verify'])
@pytest.mark.parametrize('state', [JobState.queued, JobState.processing])
def test_busy_title_is_skipped_without_clearing_decisions(tmp_path, monkeypatch, kind, state):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add(title(1)); session.add(window(1))
                if kind == 'audio':
                    job = ProcessingJob(title_id=1, state=state, wordlist_version=1, trigger_source=TriggerSource.manual)
                else:
                    job = SceneJob(title_id=1, state=state, kind=SceneJobKind(kind))
                session.add(job); await session.commit()
            result = await queue.clear_and_restart_scans([1])
            assert result.busy == 1 and result.cleared == result.restarted == 0
            assert 'Skipped 1 busy title' in result.message()
            async with factory() as session:
                assert len((await session.execute(select(DetectedScene))).scalars().all()) == 1
                assert (await session.get(Title, 1)).scene_scan_status == 'scanned'
    asyncio.run(check())


def test_concurrent_restarts_leave_one_current_scan_generation(tmp_path, monkeypatch):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add(title(1)); session.add(window(1)); await session.commit()
            await asyncio.gather(queue.clear_and_restart_scans([1]), queue.clear_and_restart_scans([1]))
            async with factory() as session:
                active = (await session.execute(select(SceneJob).where(SceneJob.state == JobState.queued))).scalars().all()
                assert len(active) == 1
                assert (await session.get(Title, 1)).scene_scan_generation == active[0].id
                assert (await session.execute(select(DetectedScene))).scalars().all() == []
    asyncio.run(check())


@pytest.mark.parametrize('url,affected', [
    ('/library/shows/10/clear-and-restart-scenes', {1, 2}),
    ('/library/shows/10/season/1/clear-and-restart-scenes', {1}),
    ('/library/title/4/clear-and-restart-scenes', {4}),
])
def test_movie_show_and_season_reset_routes_are_scoped_and_refresh_ui(tmp_path, monkeypatch, url, affected):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add_all([title(1, series=10), title(2, series=10, season=2), title(3, series=20), title(4)])
                session.add_all([window(i) for i in (1, 2, 3, 4)]); await session.commit()
            app = FastAPI(); app.include_router(scenes.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post(url, data={'confirm_reset': 'true'})
                assert response.status_code == 200, response.text
                assert 'restarted scanning from zero' in response.text
                assert 'Clear scenes &amp; restart scan' in response.text
            async with factory() as session:
                remaining = set((await session.execute(select(DetectedScene.title_id))).scalars())
                assert remaining == {1, 2, 3, 4} - affected
                active = set((await session.execute(select(SceneJob.title_id))).scalars())
                assert active == affected
    asyncio.run(check())


@pytest.mark.parametrize('view,expected', [('row_view', 'title-row-4'), ('detail_view', 'title-detail-4')])
def test_title_reset_preserves_row_or_detail_htmx_target(tmp_path, monkeypatch, view, expected):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add(title(4)); session.add(window(4)); await session.commit()
            app = FastAPI(); app.include_router(scenes.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/library/title/4/clear-and-restart-scenes',
                                             data={'confirm_reset': 'true', view: 'true'})
                assert response.status_code == 200
                assert f'id="{expected}"' in response.text
                assert 'Cleared 1 scene' in response.text
    asyncio.run(check())


def test_show_reset_reports_busy_episodes_and_leaves_other_shows_untouched(tmp_path, monkeypatch):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            async with factory() as session:
                session.add_all([title(1, series=10), title(2, series=10), title(3, series=20)])
                session.add_all([window(i) for i in (1, 2, 3)])
                session.add(ProcessingJob(title_id=1, state=JobState.processing,
                    wordlist_version=1, trigger_source=TriggerSource.manual))
                await session.commit()
            app = FastAPI(); app.include_router(scenes.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/library/shows/10/clear-and-restart-scenes', data={'confirm_reset': 'true'})
                assert response.status_code == 200
                assert 'Skipped 1 busy title' in response.text
                assert 'restarted scanning from zero for 1 title' in response.text
            async with factory() as session:
                assert set((await session.execute(select(DetectedScene.title_id))).scalars()) == {1, 3}
                assert set((await session.execute(select(SceneJob.title_id))).scalars()) == {2}
    asyncio.run(check())


def test_missing_title_reset_returns_not_found_without_queueing(tmp_path, monkeypatch):
    async def check():
        async with database(tmp_path, monkeypatch) as (factory, queue):
            app = FastAPI(); app.include_router(scenes.router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                assert (await client.post('/library/title/999/clear-and-restart-scenes', data={'confirm_reset': 'true'})).status_code == 404
            assert queue._pending.empty()
    asyncio.run(check())


@pytest.mark.parametrize('url', [
    '/library/title/1/clear-and-restart-scenes',
    '/library/shows/10/clear-and-restart-scenes',
    '/library/shows/10/season/1/clear-and-restart-scenes',
])
def test_destructive_reset_requires_explicit_confirmation(url, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError('Unconfirmed reset must not reach the queue/database')
    monkeypatch.setattr(scenes.scene_job_queue, 'clear_and_restart_scans', forbidden)
    app = FastAPI(); app.include_router(scenes.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            assert (await client.post(url)).status_code == 400
    asyncio.run(check())
