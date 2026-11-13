"""No original/audio-only publication races; preserve explicit human exceptions."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.scenes import protection, pipeline
from app.db.models import DetectedScene, SceneJob, SceneJobKind, JobState
from app.domain import SceneReviewStatus
from test_protection import database, evidence


@pytest.mark.parametrize("ready", [False,True])
def test_audio_retains_clean_master_but_never_publishes_before_visual_qualification(tmp_path,monkeypatch,ready):
    from app import external_edits
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            destination=tmp_path/'output'/'clean.mkv';destination.parent.mkdir();destination.write_bytes(b"last-good")
            if not ready:
                title.scene_analysis=None;await session.commit()
            blurs=[]
            async def process(session,**kwargs):kwargs['output_path'].write_bytes(b"audio-only");return SimpleNamespace()
            async def cache(staged,target):target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b"new-clean-audio")
            async def blur(session,title,job_id,**kwargs):
                assert kwargs['audio_source_path'].is_file() and kwargs['commit'] is False
                blurs.append(job_id);destination.write_bytes(b"qualified")
            monkeypatch.setattr(external_edits,'process_video',process)
            monkeypatch.setattr(external_edits,'_cache_audio',cache)
            monkeypatch.setattr(external_edits,'apply_scene_blur',blur)
            await external_edits.process_external_video(session,title=title,job_id=22,destination=destination)
            assert destination.read_bytes()==(b"qualified" if ready else b"last-good")
            assert bool(blurs)==ready
            assert Path(title.clean_audio_path).read_bytes()==b"new-clean-audio"
            assert Path(title.video_path).read_bytes()==b"original"
            assert not list(destination.parent.glob('*.spf-tmp*'))
    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False,True])
def test_restart_recovers_handoff_without_starting_scans_or_retrying_failed_renders(tmp_path,monkeypatch,failed):
    from app.queue import scene_worker
    async def scenario():
        async with database(tmp_path) as (session,title,factory):
            if failed:
                session.add(SceneJob(title_id=1,kind=SceneJobKind.blur,state=JobState.failed));await session.commit()
            @asynccontextmanager
            async def sessions():
                async with factory() as other:yield other
            queued=[]
            async def enqueue(title_id):queued.append(title_id);return True
            monkeypatch.setattr(scene_worker,'get_session',sessions)
            queue=scene_worker.SceneJobQueue()
            monkeypatch.setattr(queue,'enqueue_blur_if_not_already_active',enqueue)
            await queue._recover_protection_handoffs()
            assert queued==([] if failed else [1])
    asyncio.run(scenario())


@pytest.mark.parametrize('provider',['sonarr','radarr','bazarr'])
def test_new_arrival_webhooks_prepare_both_passes_without_legacy_trigger_flags(tmp_path,monkeypatch,provider):
    from app.routers import webhooks
    from app.db.session import set_setting
    async def scenario():
        async with database(tmp_path) as (session,title,factory):
            title.scene_analysis=None
            for key in ('sonarr_url','sonarr_api_key','radarr_url','radarr_api_key'):
                await set_setting(session,key,'')
            await set_setting(session,'trigger_sonarr_radarr_enabled',False)
            await set_setting(session,'trigger_bazarr_enabled',False);await session.commit()
            @asynccontextmanager
            async def sessions():
                async with factory() as other:yield other
            async def token(*args):pass
            async def upsert(other,**kwargs):return await other.get(type(title),1)
            calls=[]
            async def scan(*args):calls.append('scan');return True
            async def audio(*args):calls.append('audio')
            async def body():return {}
            parsed=dict(display_name='Synthetic',video_path=title.video_path,series_id=1,episode_id=1,
                series_title='Synthetic',season_number=1,episode_number=1,movie_id=1,poster_url=None,imdb_id=None)
            monkeypatch.setattr(webhooks,'get_session',sessions)
            monkeypatch.setattr(webhooks,'_check_token',token)
            monkeypatch.setattr(webhooks,'upsert_title',upsert)
            monkeypatch.setattr(webhooks.scene_job_queue,'enqueue_scan_if_not_already_active',scan)
            monkeypatch.setattr(webhooks,'enqueue_if_not_already_active',audio)
            monkeypatch.setattr(webhooks,'find_subtitle_for_video',lambda *args:tmp_path/'source.srt')
            monkeypatch.setattr(webhooks.sonarr_integration,'parse_import_webhook',lambda *args:parsed)
            monkeypatch.setattr(webhooks.radarr_integration,'parse_import_webhook',lambda *args:parsed)
            monkeypatch.setattr(webhooks.bazarr_integration,'parse_subtitle_downloaded_event',lambda *args:parsed)
            request=SimpleNamespace(json=body,query_params={})
            result=await getattr(webhooks,provider+'_webhook')(request)
            assert result['status']=='queued' and calls==['scan','audio']
    asyncio.run(scenario())


@pytest.mark.parametrize('subtitle',[False,True])
def test_missing_audio_is_automatically_prepared_when_subtitles_are_available(tmp_path,monkeypatch,subtitle):
    from app import library
    from app.queue import scene_worker
    from app.integrations import subtitle_lookup
    async def scenario():
        async with database(tmp_path) as (session,title,factory):
            Path(title.clean_audio_path).unlink()
            @asynccontextmanager
            async def sessions():
                async with factory() as other:yield other
            queued=[]
            async def enqueue(title_id,trigger):queued.append(title_id)
            monkeypatch.setattr(scene_worker,'get_session',sessions)
            monkeypatch.setattr(library,'enqueue_if_not_already_active',enqueue)
            monkeypatch.setattr(subtitle_lookup,'find_subtitle_for_video',lambda *args:tmp_path/'source.srt' if subtitle else None)
            await scene_worker.SceneJobQueue()._recover_protection_handoffs()
            assert queued==([1] if subtitle else [])
    asyncio.run(scenario())


@pytest.mark.parametrize('protected',[False,True])
def test_long_protection_scan_remains_cancellable(tmp_path,monkeypatch,protected):
    from app.queue import scene_worker
    from app.db.session import set_setting
    async def scenario():
        async with database(tmp_path) as (session,title,factory):
            job=await session.get(SceneJob,1);job.state=JobState.processing;job.progress_percent=99
            await set_setting(session,'scene_protection_enabled',protected);await session.commit()
            @asynccontextmanager
            async def sessions():
                async with factory() as other:yield other
            monkeypatch.setattr(scene_worker,'get_session',sessions)
            queue=scene_worker.SceneJobQueue();cancelled=[]
            queue._running[1]=SimpleNamespace(cancel=lambda:cancelled.append(True))
            accepted,_=await queue.cancel_job(1)
            assert accepted==protected and bool(cancelled)==protected
    asyncio.run(scenario())


def test_human_provenance_retains_observations():
    from app.routers.scenes import _mark_human
    row=SimpleNamespace(detector_evidence=json.dumps({'automatic_decision':True,'windows':[{'start':1,'end':2}]}))
    _mark_human(row)
    meta=json.loads(row.detector_evidence)
    assert meta['automatic_decision'] is False and meta['human_reviewed'] is True
    assert meta['windows']==[{'start':1,'end':2}]


def test_output_frame_fallback_does_not_reacquire_parent_heavy_lock(tmp_path,monkeypatch):
    from app.vision import classifier
    from app.queue import heavy_pipeline_lock
    async def scenario():
        lock=asyncio.Lock()
        monkeypatch.setattr(heavy_pipeline_lock,'HEAVY_PIPELINE_LOCK',lock)
        monkeypatch.setattr(classifier,'remote_extract_enabled',lambda:True)
        monkeypatch.setattr(classifier,'_classifier_urls',lambda:[])
        async def extract(binary,source,interval,directory,duration,**kwargs):
            assert lock.locked()
            (directory/'frame_000001.jpg').write_bytes(b'synthetic')
        async def detect(*args,**kwargs):return []
        monkeypatch.setattr(classifier,'_extract_frames_continuous',extract)
        monkeypatch.setattr(classifier,'_detect',detect)
        async with lock:
            scores=await asyncio.wait_for(classifier.scan_video_frames(tmp_path/'staged','ffmpeg',1,1,
                local_gate_owned=True),timeout=2)
        assert len(scores)==1 and scores[0].detections==[]
    asyncio.run(scenario())


def test_manual_rejections_allow_and_audio_exceptions_survive_and_control_automatic_render(tmp_path,monkeypatch):
    from app import output
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            job=await session.get(SceneJob,1);job.state=JobState.processing
            for i,options in enumerate([
                dict(start_seconds=1,end_seconds=3,status=SceneReviewStatus.rejected),
                dict(start_seconds=3,end_seconds=4,status=SceneReviewStatus.approved,action_override='allow'),
                dict(start_seconds=5,end_seconds=6,status=SceneReviewStatus.approved,audio_override=False)]):
                session.add(DetectedScene(id=10+i,title_id=1,peak_confidence=0,source='manual',**options))
            await session.commit()
            analysis=json.loads(title.scene_analysis)
            auto=DetectedScene(id=100,title_id=1,scan_generation=1,start_seconds=0,end_seconds=8,
                category_evidence=json.dumps(evidence()),detector_evidence=json.dumps({'automatic_decision':True,'protection_conservative':True}),
                peak_confidence=1,status=SceneReviewStatus.approved,source='detected')
            await pipeline._activate_scan(session,1,1,[auto],analysis=analysis)
            rows=(await session.execute(select(DetectedScene))).scalars().all()
            assert {r.id for r in rows}=={10,11,12,100}
            seen=[]
            async def destination(*args):return tmp_path/'clean.mkv'
            async def findings(**kwargs):
                assert kwargs['ignored']==[(1.,3.),(3.,4.)]
                return []
            async def probe(*args):return {'format':{'duration':'8'}}
            async def apply(**kwargs):
                seen.append(kwargs)
                await kwargs['validate_output'](tmp_path/'staged')
                await kwargs['before_publish']()
                return tmp_path/'clean.mkv'
            monkeypatch.setattr(output,'title_edit_destination',destination)
            monkeypatch.setattr(protection,'output_findings',findings)
            monkeypatch.setattr(pipeline,'probe',probe)
            monkeypatch.setattr(pipeline,'apply_blur',apply)
            monkeypatch.setattr(pipeline,'app_settings',SimpleNamespace(data_dir=tmp_path,media_root=tmp_path,ffmpeg_bin='ffmpeg',ffprobe_bin='ffprobe'))
            await pipeline.apply_scene_blur(session,title,22)
            assert seen
            assert not any(w.start<4 and w.end>1 for w in seen[0]['blur_intervals'])
            assert not any(w.start<6 and w.end>5 for w in seen[0]['mute_intervals'])
    asyncio.run(scenario())
