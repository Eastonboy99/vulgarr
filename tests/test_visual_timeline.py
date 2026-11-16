"""Visual coverage must follow verified video packets, not audio/mux duration."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.vision import timeline, contextual
from app.scenes.protection import validate_detector_coverage, ProtectionNotReady
from app.scenes.content_policy import validate_evidence


def media(duration=10):
    return {'format':{'duration':str(duration),'start_time':'0'},'streams':[
        {'index':0,'codec_type':'video','start_time':'0','avg_frame_rate':'25/1'},
        {'index':1,'codec_type':'audio','duration':str(duration)}]}


def test_real_title_3807_regression_no_internal_gap_and_audio_mux_tail(monkeypatch):
    info=media(7455.144)
    info['streams'][0].update(avg_frame_rate='24000/1001',tags={'DURATION':'02:04:15.031000000'})
    calls=[]
    async def run(cmd,**kwargs):
        calls.append(cmd)
        return 0,json.dumps({'packets':[{'pts_time':'7454.989','duration_time':'0.041'}]}),''
    monkeypatch.setattr(timeline,'_run',run)
    result=asyncio.run(timeline.probe_visual_timeline('ffprobe','source',info))
    assert result.container_duration==7455.144 and result.video_end==pytest.approx(7455.03)
    assert result.last_sample_time<7454.989
    scores=[SimpleNamespace(timestamp=i*.25,detections=[]) for i in range(29820)]
    with pytest.raises(ProtectionNotReady,match='tail gap 0.394s'):
        validate_detector_coverage(scores,result.container_duration,.25)
    validate_detector_coverage(scores,result.video_end,.25)
    broken=list(scores);broken.pop(500)
    with pytest.raises(ProtectionNotReady,match='timestamp gaps'):
        validate_detector_coverage(broken,result.video_end,.25)
    assert '-select_streams' in calls[0] and '-read_intervals' in calls[0]
    # Seek in the tail but read to EOF, never infer EOF from a fixed interval.
    assert calls[0][calls[0].index('-read_intervals')+1]=='7395.144000%'


@pytest.mark.parametrize('case',['short_tag','audio_tail','origin','attached_pic','missing_packet_duration'])
def test_video_horizon_is_verified_not_blindly_taken_from_metadata(monkeypatch,case):
    info=media(100 if case=='audio_tail' else 10)
    info['streams'][0]['tags']={'DURATION':'00:00:05.000'}
    if case=='audio_tail':info['streams'][0]['tags']['DURATION']='00:00:10.000'
    if case=='origin':
        info['format']['start_time']='5';info['streams'][0]['start_time']='5'
    if case=='attached_pic':
        info['streams'][0]['index']=2
        info['streams'].insert(0,{'index':0,'codec_type':'video','disposition':{'attached_pic':1}})
    calls=[]
    async def run(cmd,**kwargs):
        calls.append(cmd)
        packets=[] if case=='audio_tail' and len(calls)==1 else [
            {'pts_time':'14.96' if case=='origin' else '9.96',
             **({} if case=='missing_packet_duration' else {'duration_time':'0.04'})}]
        return 0,json.dumps({'packets':packets}),''
    monkeypatch.setattr(timeline,'_run',run)
    result=asyncio.run(timeline.probe_visual_timeline('ffprobe','source',info))
    assert result.video_end==pytest.approx(10)
    assert result.last_sample_time==pytest.approx(9.959)
    assert len(calls)==(2 if case=='audio_tail' else 1)
    if case=='attached_pic':assert calls[0][calls[0].index('-select_streams')+1]=='2'
    if case=='origin':assert calls[0][calls[0].index('-read_intervals')+1]=='5.000000%'


def test_long_held_final_frame_does_not_seek_back_to_an_earlier_image(monkeypatch):
    async def run(*args,**kwargs):
        return 0,json.dumps({'packets':[
            {'pts_time':'9.96','duration_time':'0.04'},
            {'pts_time':'10.0','duration_time':'10.0'}]}),''
    monkeypatch.setattr(timeline,'_run',run)
    result=asyncio.run(timeline.probe_visual_timeline('ffprobe','source',media(20)))
    assert result.video_end==20
    assert 9.96<result.last_sample_time<10


@pytest.mark.parametrize('case',['no_video','nonfinite','delayed_video','probe_error','bad_json','no_packets','short_container','bad_packet_schema'])
def test_unverifiable_horizons_fail_closed(monkeypatch,case):
    info=media(10)
    if case=='no_video':info['streams']=[]
    if case=='nonfinite':info['format']['duration']='nan'
    if case=='delayed_video':info['streams'][0]['start_time']='5'
    calls=[]
    async def run(cmd,**kwargs):
        calls.append(cmd)
        if case=='probe_error':return 1,'','private provider error'
        if case=='bad_json':return 0,'invalid json',''
        packets=[] if case=='no_packets' else [
            {'pts_time':'19.96' if case=='short_container' else '9.96','duration_time':'0.04'}]
        if case=='bad_packet_schema':packets=[None]
        return 0,json.dumps({'packets':packets}),''
    monkeypatch.setattr(timeline,'_run',run)
    with pytest.raises(ValueError) as error:
        asyncio.run(timeline.probe_visual_timeline('ffprobe','source',info))
    assert 'private provider error' not in str(error.value)


def test_last_partial_context_window_samples_the_last_decodable_frame():
    assert contextual.sample_timestamps(0,8,7.98)==[1,3,5,7]
    assert contextual.sample_timestamps(8,8.02,7.98)==[7.98]*4


@pytest.mark.parametrize('times',[[0,1,2],[0,2,1,3],[0,1,2,float('nan')],[-1,0,1,2]])
def test_invalid_backend_sample_timestamps_are_not_trusted(times):
    evidence=contextual.observations_to_evidence({'frames':[{'frame_id':f'f{i}','state':'clothed','evidence':'Synthetic'} for i in range(4)],'window_summary':'Synthetic'},'mock')
    evidence['sample_timestamps']=times
    with pytest.raises(ValueError,match='sample timestamps'):validate_evidence(evidence)
