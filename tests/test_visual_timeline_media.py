"""Opt-in real FFmpeg proof, using synthetic media in isolated worker processes."""
import inspect
import json
import os
from pathlib import Path
import subprocess

import pytest

from app.scenes.protection import validate_detector_coverage
from app.vision import timeline

CONTAINERS = [c.strip() for c in os.getenv('SPF_TEST_RENDER_CONTAINERS', '').split(',') if c.strip()]
pytestmark = pytest.mark.skipif(not CONTAINERS, reason='Opt-in real Docker timeline qualification')

DIAGNOSTIC = r'''
import asyncio,json,math,subprocess,sys,tempfile,types
from pathlib import Path
from types import SimpleNamespace
request=json.load(sys.stdin)
def execute(command):
    result=subprocess.run(command,capture_output=True,timeout=60)
    assert result.returncode==0, result.stderr.decode(errors='replace')[-500:]
    return result.stdout
async def run(command,**kwargs):
    return 0,execute(command).decode(),''
async def probe(binary,path):
    return json.loads(execute([binary,'-v','error','-show_streams','-show_format','-of','json',str(path)]))
for name in ('app','app.mux','app.mux.remux'):
    sys.modules[name]=types.ModuleType(name)
sys.modules['app.mux.remux']._run=run
sys.modules['app.mux.remux'].probe=probe
module=types.ModuleType('isolated_timeline');sys.modules[module.__name__]=module
exec(request['timeline'],module.__dict__)
scope={'math':math,'ProtectionNotReady':ValueError}
exec(request['coverage'],scope)
with tempfile.TemporaryDirectory(prefix='vulgarr-timeline-diagnostic-') as folder:
    source=Path(folder)/'source.mkv'
    execute(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
        'testsrc2=size=64x64:rate=24:duration=1','-f','lavfi','-i',
        'sine=sample_rate=48000:duration=1.14','-c:v','libx264','-c:a','aac',str(source)])
    measured=asyncio.run(module.probe_visual_timeline('ffprobe',source))
    scores=[SimpleNamespace(timestamp=i*.25,detections=[]) for i in range(4)]
    scope['validate_detector_coverage'](scores,measured.video_end,.25)
    try:scope['validate_detector_coverage'](scores,measured.container_duration,.25)
    except ValueError:pass
    else:raise AssertionError('Fixture did not reproduce the old false tail failure')
    exists={}
    for name,t in {'after_last_pts':measured.video_end-.001,'safe':measured.last_sample_time}.items():
        image=Path(folder)/(name+'.jpg')
        execute(['ffmpeg','-nostdin','-v','error','-ss',f'{t:.6f}','-i',str(source),'-frames:v','1',str(image)])
        exists[name]=image.is_file() and image.stat().st_size>0
    assert not exists['after_last_pts'] and exists['safe'], exists
    print(json.dumps({'synthetic_only':True,'container_duration':measured.container_duration,
        'verified_video_end':measured.video_end,'safe_final_frame_extracted':exists['safe']}))
'''


@pytest.mark.parametrize('container', CONTAINERS or ['not-configured'])
def test_real_audio_tail_and_final_frame_seek(container):
    payload={'timeline':Path(timeline.__file__).read_text(encoding='utf-8'),
             'coverage':inspect.getsource(validate_detector_coverage)}
    result=subprocess.run(['docker','exec','-i',container,'python3','-c',DIAGNOSTIC],
        input=json.dumps(payload).encode(),capture_output=True,timeout=120)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-2000:]
    assert json.loads(result.stdout)['safe_final_frame_extracted']
