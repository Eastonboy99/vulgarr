"""Protection-first automation with real transactional DB and synthetic evidence."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import Base, Title, DetectedScene, SceneJob, SceneJobKind, JobState, MediaType
from app.db.session import set_setting, get_setting
from app.domain import SceneReviewStatus
from app.scenes import protection, pipeline
from app.scenes.content_policy import DEFAULT_POLICY, Action, scene_decision
from app.vision import contextual


def evidence(state="contextual_nudity"):
    return contextual.observations_to_evidence({"frames": [{"frame_id": f"f{i}", "state": state,
        "evidence": "Synthetic visual observation"} for i in range(4)], "window_summary": "Synthetic summary"}, "mock")


@asynccontextmanager
async def database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'protection.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        source, audio = tmp_path/'source.mkv', tmp_path/'clean.mka'
        source.write_bytes(b"original"); audio.write_bytes(b"clean-audio")
        title = Title(id=1, media_type=MediaType.movie, display_name="Synthetic", video_path=str(source),
            clean_audio_path=str(audio), clean_audio_source_fingerprint=protection.source_fingerprint(source),
            scene_scan_status="scanned", scene_scan_generation=1)
        session.add(title)
        session.add(SceneJob(id=1, title_id=1, kind=SceneJobKind.scan, state=JobState.done))
        for key,value in {"scene_protection_enabled": True, "scene_contextual_base_url": "http://localhost:1234/v1",
                          "scene_contextual_model": "mock", "movie_output_folder": str(tmp_path/'output')}.items():
            await set_setting(session,key,value)
        await session.commit()
        await stamp(session,title)
        try:
            yield session,title,factory
        finally:
            await session.close()
    await engine.dispose()


async def stamp(session,title):
    title.scene_analysis = json.dumps({"protection": {"fingerprint": await protection.fingerprint(session),
        "source": protection.source_fingerprint(Path(title.video_path)), "generation": title.scene_scan_generation,
        "complete": True}})
    await session.commit()


@pytest.mark.parametrize("action", ["allow","blur","blur_mute"])
@pytest.mark.parametrize("state", ["contextual_nudity","uncertain"])
@pytest.mark.parametrize("mute", [False,True])
def test_conservative_decisions_follow_policy_without_inventing_labels(action,state,mute):
    policy=json.loads(json.dumps(DEFAULT_POLICY));policy["actions"]["nudity.contextual"]=action
    policy["mute_blurred_audio"]=mute
    raw=evidence(state)
    row=SimpleNamespace(category_evidence=json.dumps(raw),detector_evidence=json.dumps({"protection_conservative":True}),
        action_override=None,audio_override=None,mute_audio=False)
    assert scene_decision(row,policy).action.value == ("blur_mute" if action=="blur" and mute else action)
    assert bool(raw["labels"]) == (state == "contextual_nudity")
    row.action_override="allow"
    assert scene_decision(row,policy).action == Action.allow
    row.action_override=None;row.audio_override=False
    if action!="allow":assert scene_decision(row,policy).action == Action.blur


@pytest.mark.parametrize("failure", [None,"missing","incomplete","generation","source","policy","model","pending","failed_scan"])
def test_publication_gate_rejects_incomplete_stale_and_unresolved_work(tmp_path,failure):
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            manifest=json.loads(title.scene_analysis)
            if failure=="missing":manifest={}
            if failure=="incomplete":manifest["protection"]["complete"]=False
            if failure=="generation":title.scene_scan_generation=2
            if failure=="source":Path(title.video_path).write_bytes(b"changed original")
            if failure=="policy":
                policy=json.loads(json.dumps(DEFAULT_POLICY));policy["actions"]["nudity.contextual"]="allow"
                await set_setting(session,"scene_content_policy",policy)
            if failure=="model":await set_setting(session,"scene_contextual_model","other")
            if failure=="pending":session.add(DetectedScene(title_id=1,start_seconds=1,end_seconds=2,
                peak_confidence=0,status=SceneReviewStatus.pending,source="manual"))
            if failure=="failed_scan":title.scene_scan_status="scan_failed"
            title.scene_analysis=json.dumps(manifest);await session.commit()
            if failure:
                with pytest.raises(protection.ProtectionNotReady):await protection.readiness(session,title)
            else:
                await protection.readiness(session,title)
                assert await protection.audio_ready(session,title)
            assert Path(title.video_path).read_bytes() in (b"original",b"changed original")
    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False,True])
def test_shot_changes_include_short_shots_or_fail_closed(monkeypatch,failed):
    from app.mux import remux
    async def run(*args,**kwargs):return (1 if failed else 0,"frame:0 pts:2 pts_time:2.125\nframe:1 pts:3 pts_time:2.250\nframe:2 pts:9 pts_time:9.9","")
    monkeypatch.setattr(remux,"_run",run)
    if failed:
        with pytest.raises(protection.ProtectionNotReady):asyncio.run(protection.shot_windows(Path("synthetic"),10,"ffmpeg"))
    else:
        assert asyncio.run(protection.shot_windows(Path("synthetic"),10,"ffmpeg")) == [(2.125,2.25),(2.25,3.0),(9.9,10)]


def test_source_rechecks_do_not_veto_positives_and_expand_boundaries(tmp_path,monkeypatch):
    calls=[]
    async def shots(*args):return [(1,1.5)]
    async def scan(**kwargs):
        calls.append(kwargs)
        if kwargs.get("purpose")=="recheck":
            if kwargs.get("max_long_edge")==1024:
                return [(a,b,evidence("clothed")) for a,b in kwargs["window_plan"]]
            return [(a,b,evidence("uncertain")) for a,b in kwargs["window_plan"]]
        return [(0,4,evidence()),(4,8,evidence("uncertain"))]
    monkeypatch.setattr(protection,"shot_windows",shots)
    monkeypatch.setattr(contextual,"scan_contextual_windows",scan)
    config={"scene_contextual_base_url":"http://localhost/v1","scene_contextual_api_key":"",
        "scene_contextual_model":"mock","scene_contextual_window_seconds":4,"scene_contextual_timeout_seconds":300}
    windows,coverage=asyncio.run(protection.analyze_windows(video_path=Path("source"),duration=8,
        checkpoint_dir=tmp_path,fingerprint="source",config=config,policy=DEFAULT_POLICY))
    assert coverage["complete"] and coverage["uncertainty_rechecks"]==1
    assert calls[0]["extra_windows"]==[(1,1.5)]
    assert windows[0][2]["labels"]=={"nudity.contextual":1.0}
    assert any(k["window_plan"]==[(4,6.0)] for k in calls if "window_plan" in k)


@pytest.mark.parametrize("state,covered,ignored,count", [
    ("uncertain",[(0,4)],[],0),("uncertain",[],[],1),
    ("contextual_nudity",[(0,4)],[],1),("contextual_nudity",[],[(1,3)],2),
    ("clothed",[],[],0)])
def test_output_checks_distinguish_expected_blur_from_residual_content(tmp_path,monkeypatch,state,covered,ignored,count):
    from app.vision import classifier
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            async def shots(*args,**kwargs):return []
            async def scores(*args,**kwargs):
                assert kwargs['start_offset']==.125 and kwargs['local_gate_owned'] is True
                return [SimpleNamespace(timestamp=.125+i*.25,confidence=0,detections=[]) for i in range(32)]
            async def scan(**kwargs):
                assert kwargs["purpose"]=="output"
                assert (0.,.5) in kwargs["window_plan"] and (.5,4.5) in kwargs["window_plan"]
                return [(0,4,evidence(state))]
            monkeypatch.setattr(protection,"shot_windows",shots)
            monkeypatch.setattr(classifier,"scan_video_frames",scores)
            monkeypatch.setattr(contextual,"scan_contextual_windows",scan)
            rendered=tmp_path/'rendered.mkv';rendered.write_bytes(b"synthetic output")
            result=await protection.output_findings(session=session,title=title,path=rendered,
                work_dir=tmp_path/'check',policy=DEFAULT_POLICY,duration=8,covered=covered,ignored=ignored)
            assert len(result)==count
            if ignored:assert [(a,b) for a,b,_ in result]==[(0,1),(3,4)]
    asyncio.run(scenario())


@pytest.mark.parametrize("residual", [False,True])
def test_low_level_output_check_happens_before_publish_without_encoder_retry(tmp_path,monkeypatch,residual):
    from app.mux import scene_blur
    source,final,staged=tmp_path/'source.mkv',tmp_path/'clean.mkv',tmp_path/'staged.mkv'
    source.write_bytes(b"original");final.write_bytes(b"last-good")
    calls=[]
    async def probe(*args):return {"streams":[],"format":{"duration":"8"}}
    async def build(**kwargs):calls.append(1);staged.write_bytes(b"new");return staged
    async def structural(**kwargs):pass
    async def content(path):
        assert path==staged and final.read_bytes()==b"last-good"
        if residual:raise protection.ResidualContentError([(0,1,evidence())])
    monkeypatch.setattr(scene_blur,"probe",probe)
    monkeypatch.setattr(scene_blur,"build_blurred_video",build)
    monkeypatch.setattr(scene_blur,"verify_blurred_output",structural)
    args=dict(video_path=source,output_path=final,blur_intervals=[],mute_intervals=[],ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",video_crf=18,video_preset="medium",work_dir=tmp_path/'work',validate_output=content)
    if residual:
        with pytest.raises(protection.ResidualContentError):asyncio.run(scene_blur.apply_blur(**args))
        assert final.read_bytes()==b"last-good" and not staged.exists()
    else:
        asyncio.run(scene_blur.apply_blur(**args));assert final.read_bytes()==b"new"
    assert calls==[1] and source.read_bytes()==b"original"


@pytest.mark.parametrize("mode", ["clean","corrected","persistent","stale","cancelled","missing_destination"])
def test_automatic_render_is_bounded_and_preserves_previous_output(tmp_path,monkeypatch,mode):
    from app import output
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            final=tmp_path/'published.mkv';final.write_bytes(b"last-good")
            if mode=='cancelled':
                session.add(SceneJob(id=22,title_id=1,kind=SceneJobKind.blur,state=JobState.cancelled))
                await session.commit()
            calls=[]
            async def destination(*args):return None if mode=='missing_destination' else final
            async def probe(*args):return {"format":{"duration":"8"}}
            async def findings(**kwargs):
                if mode=="persistent" or (mode=="corrected" and len(calls)==1):return [(1,2,evidence())]
                return []
            async def apply(**kwargs):
                calls.append(kwargs)
                await kwargs["validate_output"](tmp_path/'staged')
                if mode=="stale":await set_setting(session,"scene_contextual_model","changed")
                await kwargs["before_publish"]()
                final.write_bytes(b"qualified");return final
            monkeypatch.setattr(output,"title_edit_destination",destination)
            monkeypatch.setattr(pipeline,"probe",probe)
            monkeypatch.setattr(pipeline,"apply_blur",apply)
            monkeypatch.setattr(protection,"output_findings",findings)
            monkeypatch.setattr(pipeline,"app_settings",SimpleNamespace(data_dir=tmp_path,media_root=tmp_path,ffmpeg_bin="ffmpeg",ffprobe_bin="ffprobe"))
            if mode in ("persistent","stale","cancelled","missing_destination"):
                with pytest.raises(ValueError):await pipeline.apply_scene_blur(session,title,22)
                assert final.read_bytes()==b"last-good"
            else:
                result=await pipeline.apply_scene_blur(session,title,22)
                assert result.output_path==final and final.read_bytes()==b"qualified"
            assert len(calls)==(0 if mode=='missing_destination' else 2 if mode in ("corrected","persistent") else 1)
            if mode=="corrected":
                assert calls[1]["blur_radius"]>=90 and calls[1]["mute_intervals"]
                rows=(await session.execute(select(DetectedScene))).scalars().all()
                assert len(rows)==1 and rows[0].status==SceneReviewStatus.approved
                assert json.loads(rows[0].detector_evidence)["backend"]=="local-output-correction"
            assert Path(title.video_path).read_bytes()==b"original"
    asyncio.run(scenario())


@pytest.mark.parametrize("changed", [False,True])
def test_whole_title_auto_approval_and_configuration_fencing(tmp_path,monkeypatch,changed):
    async def scenario():
        async with database(tmp_path) as (session,title,_):
            job=await session.get(SceneJob,1);job.state=JobState.processing;await session.commit()
            async def local(*args):return None
            async def probe(*args):return {"format":{"duration":"8"}}
            async def scores(*args,**kwargs):
                assert kwargs["frame_interval_seconds"]==.25
                return [SimpleNamespace(timestamp=i*.25,confidence=0,detections=[]) for i in range(32)]
            async def analyze(**kwargs):
                assert kwargs["config"]["scene_contextual_window_seconds"]==4
                if changed:await set_setting(session,"scene_contextual_model","changed")
                return [(0,4,evidence()),(4,8,evidence("uncertain"))],{"complete":True}
            monkeypatch.setattr(contextual,"local_endpoint",local)
            monkeypatch.setattr(pipeline,"probe",probe)
            monkeypatch.setattr(pipeline,"scan_video_frames",scores)
            monkeypatch.setattr(protection,"analyze_windows",analyze)
            monkeypatch.setattr(pipeline,"app_settings",SimpleNamespace(data_dir=tmp_path,ffmpeg_bin="ffmpeg",ffprobe_bin="ffprobe"))
            if changed:
                with pytest.raises(ValueError):await pipeline.scan_for_scenes(session,title,1)
                assert not (await session.execute(select(DetectedScene))).scalars().all()
            else:
                outcome=await pipeline.scan_for_scenes(session,title,1)
                assert outcome.auto_approved_count==2 and outcome.need_review_count==0
                rows=(await session.execute(select(DetectedScene))).scalars().all()
                assert all(r.status==SceneReviewStatus.approved for r in rows)
                assert all(scene_decision(r,DEFAULT_POLICY).action!=Action.allow for r in rows)
                assert json.loads(title.scene_analysis)["protection"]["complete"]
                await protection.readiness(session,title)
                from app.routers.scenes import _scene_review_context
                view=await _scene_review_context(session,1)
                assert not view['needs_reanalysis'] and not view['claude_vision_verify_enabled']
                assert all(r.protection_automatic for r in view['scenes'])
    asyncio.run(scenario())
