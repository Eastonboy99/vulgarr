"""Real synthetic FFmpeg + host cache proof; never loads a real detector.

The existing workers lack Pillow (the new application dependency), so FFmpeg
runs there and synthetic JPEGs cross a test adapter into the host's real cache.
Both the production extraction function and worker snapshot function execute
unchanged in the isolated worker process; no running service is modified.
"""
import ast
import asyncio
import base64
import inspect
import json
import os
from pathlib import Path
import subprocess

import pytest
from PIL import Image
import io
from app.vision import classifier,frame_cache
from remote_worker import inference

CONTAINERS=[c.strip() for c in os.getenv('SPF_TEST_RENDER_CONTAINERS','').split(',') if c.strip()]
pytestmark=pytest.mark.skipif(not CONTAINERS,reason='Opt-in synthetic Docker frame-cache proof')

DIAGNOSTIC=r'''
import asyncio,json,subprocess,sys,tempfile,threading,math,base64
from pathlib import Path
from types import SimpleNamespace
request=json.load(sys.stdin)
extract={'asyncio':asyncio,'Path':Path};exec(request['extract'],extract)
guard={};exec(request['guard'],guard)
scan={'ScanRequest':SimpleNamespace,'validate_source':lambda source:None,
      'math':math,'tempfile':tempfile,'Path':Path,'subprocess':subprocess,'base64':base64,
      '_decode_lock':threading.Lock(),'_EXPLICIT':set(),
      '_inference':guard['InferenceGuard'](),'detector':lambda:SimpleNamespace(detect=lambda path:[])}
exec(request['scan'],scan)
async def run(folder):
    source=folder/'source.mkv'
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
        'testsrc2=size=1920x800:rate=24:duration=2','-c:v','libx264','-preset','ultrafast',str(source)],capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-500:]
    frames=folder/'fallback';frames.mkdir()
    await extract['_extract_frames_continuous']('ffmpeg',source,.25,frames,2,start_offset=0,clip_duration=2)
    fallback=[base64.b64encode(p.read_bytes()).decode() for p in sorted(frames.glob('frame_*.jpg'))]
    scores,samples=scan['_scan_chunk'](SimpleNamespace(source_url=str(source),start=0,duration=2,interval=.25,include_samples=True))
    assert len(scores)==8 and len(samples)==2 and len(fallback)==8
    print(json.dumps({'synthetic_only':True,'scores':scores,'samples':samples,'fallback':fallback}))
with tempfile.TemporaryDirectory(prefix='vulgarr-cache-test-') as folder:
    asyncio.run(run(Path(folder)))
'''


@pytest.mark.parametrize('container',CONTAINERS or ['not-configured'])
def test_shared_decode_and_cache_fallback_with_real_ffmpeg(container,monkeypatch,tmp_path):
    source=Path('remote_worker/server.py').read_text(encoding='utf-8')
    scan=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_scan_chunk')
    payload={'guard':Path(inference.__file__).read_text(encoding='utf-8'),
             'extract':inspect.getsource(classifier._extract_frames_continuous),
             'scan':ast.get_source_segment(source,scan)}
    result=subprocess.run(['docker','exec','-i',container,'python3','-c',DIAGNOSTIC],
        input=json.dumps(payload).encode(),capture_output=True,timeout=120)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-2000:]
    data=json.loads(result.stdout)
    signature=inspect.signature(classifier._extract_frames_continuous)
    calls=[]
    async def remote_extract(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        assert bound.arguments['frame_interval_seconds']==.25
        assert bound.arguments['clip_duration']==2
        calls.append(1)
        out=bound.arguments['out_dir']
        for i,image in enumerate(data['fallback']):
            (out/f'frame_{i:08d}.jpg').write_bytes(base64.b64decode(image))
    monkeypatch.setattr(classifier,'_extract_frames_continuous',remote_extract)
    async def verify():
        fallback=frame_cache.FrameCache(tmp_path/'fallback','source','synthetic','ffmpeg',2,local_gate_owned=True)
        assert await fallback.frame(.5) and await fallback.frame(1.5)
        assert fallback.fallback_chunks==1 and len(calls)==1
        assert Image.open(io.BytesIO(await fallback.frame(.5,768))).width==768
        assert Image.open(io.BytesIO(await fallback.frame(.5,1024))).width==1024
        assert await fallback.frame(.625) is None
        shared=frame_cache.FrameCache(tmp_path/'shared','source','synthetic','ffmpeg',2,local_gate_owned=True)
        shared.accept_remote({'sample_protocol':'fps-cache-v1','sample_frames':data['samples']},data['scores'])
        assert await shared.frame(.5) and await shared.frame(1.5,1024)
        assert shared.fallback_chunks==0 and len(calls)==1
    asyncio.run(verify())
