"""Opt-in, real NVDEC/boxblur/NVENC regression on running Docker workers.

SPF_TEST_RENDER_CONTAINERS=worker-0,worker-1 python -m pytest tests/test_remote_render_gpu.py -q

Candidate render.py is loaded in an isolated diagnostic Python process through
stdin. No image rebuild, daemon code modification, service restart, model load
or user media is involved. Synthetic files are removed on completion.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

CONTAINERS = [s.strip() for s in os.getenv('SPF_TEST_RENDER_CONTAINERS', '').split(',') if s.strip()]
pytestmark = pytest.mark.skipif(not CONTAINERS, reason='Opt-in real Docker GPU qualification')

_DIAGNOSTIC = r'''
import json, subprocess, sys, tempfile, types
from pathlib import Path
import numpy as np
request = json.load(sys.stdin)
module = types.ModuleType('vulgarr_render_diagnostic')
sys.modules[module.__name__] = module
exec(compile(request['module'], '<candidate-render.py>', 'exec'), module.__dict__)
codec, fmt = request['codec'], request['pix_fmt']
w, h, frames = 1280, 720, 8
bit_depth = 10 if '10' in fmt else 8

def execute(command):
    result = subprocess.run(command, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Synthetic command failed: ' + result.stderr.decode(errors='replace')[-1500:])
    return result.stdout

with tempfile.TemporaryDirectory(prefix='blur-gpu-regression-') as temp:
    source = Path(temp)/'gray.mkv'
    source_encoder = 'libx265' if codec == 'hevc' else 'libx264'
    source_options = ['-x265-params', 'pools=2:frame-threads=1'] if codec == 'hevc' else ['-threads', '2']
    execute(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
             f'color=c=gray:s={w}x{h}:r=25,format={fmt}', '-frames:v', str(frames),
             '-c:v', source_encoder, '-preset', 'ultrafast', *source_options, str(source)])
    params = module.RenderRequest(source_url=str(source), source_codec=codec,
        duration=frames/25, width=w, height=h, pix_fmt=fmt, crf=23, preset='ultrafast',
        filter_complex="[0:v]boxblur=luma_radius=90:luma_power=8:chroma_radius=90:chroma_power=8:enable='between(t,0.040,0.160)'[vb]")
    reports = []
    for cuda in [False, True]:
        output = Path(temp)/('cuda-'+str(cuda)+'.mkv')
        execute(module._command(params, output, 'NVENC', cuda))
        metadata = json.loads(execute(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output)]))
        video = metadata['streams'][0]
        assert video['codec_name'] == codec, video
        assert video['pix_fmt'] == fmt, video  # Do not hide the bug by dropping to 8-bit.
        raw = execute(['ffmpeg', '-v', 'error', '-i', str(output), '-pix_fmt', fmt, '-f', 'rawvideo', 'pipe:1'])
        pixels = np.frombuffer(raw, dtype='uint8' if bit_depth == 8 else '<u2').reshape(-1, w*h*3//2)
        assert len(pixels) == frames, len(pixels)
        expected_chroma = 1 << (bit_depth-1)
        # Every frame, both halves, both planes—including blur on/off boundaries.
        chroma = pixels[:, w*h:].astype(np.int32)
        deviation = int(np.max(np.abs(chroma-expected_chroma)))
        assert deviation <= (4 if bit_depth == 8 else 16), (cuda, fmt, deviation)
        u = pixels[2, w*h:w*h*5//4].reshape(h//2, w//2)
        reports.append({'cuda': cuda, 'pix_fmt': video['pix_fmt'], 'max_chroma_error': deviation,
                        'blurred_u_left': float(u[:, :w//4].mean()),
                        'blurred_u_right': float(u[:, w//4:].mean())})
    print(json.dumps(reports))
'''


@pytest.mark.parametrize('container', CONTAINERS or ['unconfigured'])
@pytest.mark.parametrize('codec,pix_fmt', [('h264', 'yuv420p'), ('hevc', 'yuv420p'), ('hevc', 'yuv420p10le')])
def test_real_cuda_blur_preserves_both_chroma_halves(container, codec, pix_fmt):
    module = Path(__file__).resolve().parents[1]/'remote_worker'/'render.py'
    result = subprocess.run(['docker', 'exec', '-i', container, 'python3', '-c', _DIAGNOSTIC],
        input=json.dumps({'module': module.read_text(), 'codec': codec, 'pix_fmt': pix_fmt}),
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-3000:]
    reports = json.loads(result.stdout)
    assert [r['cuda'] for r in reports] == [False, True]
    assert all(r['pix_fmt'] == pix_fmt for r in reports)
