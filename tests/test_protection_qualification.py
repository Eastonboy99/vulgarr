import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from app.scenes.qualification import interval_metrics
from app.scenes import protection


def test_metrics_measure_uncovered_seconds_not_just_any_overlap():
    result=interval_metrics([[0,10]],[[0,1],[20,22]])
    assert result['duration_recall']==.1
    assert result['missed_seconds']==9 and result['extra_filtered_seconds']==2
    assert result['missed_intervals']==[(1.,10.)]


def test_clothed_controls_measure_unnecessary_blurring():
    result=interval_metrics([],[[1,3],[2,5]])
    assert result['duration_recall'] is None and result['extra_filtered_seconds']==4
    assert result['filtered_seconds']==4


@pytest.mark.parametrize('bad',[[[1,1]],[[-1,2]],[[0,float('nan')]],[[0,True]]])
def test_invalid_annotations_are_rejected(bad):
    with pytest.raises(ValueError):interval_metrics(bad,[])


@pytest.mark.parametrize('failure',['empty','gap','raw_missing','end_missing'])
def test_incomplete_detector_sampling_is_not_a_safe_result(failure):
    from types import SimpleNamespace
    scores=[SimpleNamespace(timestamp=i*.25,detections=[]) for i in range(32)]
    if failure=='empty':scores=[]
    if failure=='gap':scores.pop(10)
    if failure=='raw_missing':scores[3].detections=None
    if failure=='end_missing':scores=scores[:-5]
    with pytest.raises(protection.ProtectionNotReady):protection.validate_detector_coverage(scores,8,.25)


@pytest.mark.parametrize('failure',[None,'types','channels','rate','sync','nan'])
def test_structural_output_checks_reject_track_and_sync_regressions(tmp_path,monkeypatch,failure):
    from app.mux import scene_blur
    from app.mux.remux import RemuxError
    original={'format':{'duration':'8','start_time':'0'},'streams':[
        {'codec_type':'video','start_time':'0'},
        {'codec_type':'audio','start_time':'0','channels':2,'sample_rate':'48000'}]}
    output=deepcopy(original)
    if failure=='types':output['streams'][1]['codec_type']='subtitle'
    if failure=='channels':output['streams'][1]['channels']=1
    if failure=='rate':output['streams'][1]['sample_rate']='24000'
    if failure=='sync':output['streams'][1]['start_time']='1'
    if failure=='nan':output['format']['duration']='nan'
    async def probe(*args):return output
    async def packets(*args):pass
    monkeypatch.setattr(scene_blur,'probe',probe)
    monkeypatch.setattr(scene_blur,'verify_packet_interleaving',packets)
    async def run():
        await scene_blur.verify_blurred_output(ffmpeg_bin='ffmpeg',ffprobe_bin='ffprobe',original_probe=original,
            tmp_path=tmp_path/'staged',blurred_windows=[])
    if failure:
        with pytest.raises(RemuxError):asyncio.run(run())
    else:asyncio.run(run())
