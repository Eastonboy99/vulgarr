"""Opt-in tone-energy proof using candidate code in isolated Docker processes.

No service changes or user media. Requires SPF_TEST_RENDER_CONTAINERS, like
existing mux/GPU diagnostics. AAC-LC stereo/7.1 fixtures are non-silent. HE-AAC
qualification still requires an authorized fixture/encoder and is not inferred
from these tests.
"""
import inspect
import json
import os
import subprocess

import pytest

from app.audio.mute import build_volume_filter
from app.mux.scene_blur import _filter_audio_continuously

CONTAINERS = [c.strip() for c in os.getenv("SPF_TEST_RENDER_CONTAINERS", "").split(",") if c.strip()]
pytestmark = pytest.mark.skipif(not CONTAINERS, reason="Opt-in real Docker audio qualification")

DIAGNOSTIC = r'''
import asyncio, json, math, subprocess, sys, tempfile, types
from pathlib import Path
from dataclasses import dataclass
import numpy as np
request = json.load(sys.stdin)
@dataclass
class MuteInterval:
    start: float
    end: float
scope = {'MuteInterval': MuteInterval, '_MAX_TERMS_PER_STAGE': 20}
exec(request['volume'], scope)
module = types.ModuleType('app.audio.mute')
module.build_volume_filter = scope['build_volume_filter']
sys.modules['app.audio.mute'] = module

def execute(command):
    result = subprocess.run(command, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    return result.stdout

async def _run_atomic(build, output, timeout, error):
    execute(build(output))

scope = {'Path': Path, 'MuteInterval': MuteInterval, '_run_atomic': _run_atomic}
exec(request['filter'], scope)

def probe(path):
    return json.loads(execute(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams']

def first_packet(path, track):
    data = json.loads(execute(['ffprobe', '-v', 'error', '-select_streams', f'a:{track}',
        '-show_packets', '-read_intervals', '%+#1', '-of', 'json', str(path)]))
    return float(data['packets'][0]['pts_time'])

with tempfile.TemporaryDirectory(prefix='vulgarr-audio-diagnostic-') as folder:
    source, output = Path(folder)/'tone.mkv', Path(folder)/'muted.mkv'
    channels = request['channels']
    layout = 'stereo' if channels == 2 else '7.1'
    tone = '|'.join(['0.2*sin(2*PI*440*t)'] * channels)
    fixture = f'aevalsrc={tone}:s=48000:d=4:c={layout}'
    execute(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', fixture,
             '-itsoffset', '0.35', '-f', 'lavfi', '-i', fixture,
             '-map', '0:a', '-map', '1:a', '-c:a', 'aac', '-b:a', '512k', str(source)])
    cases = {
        'middle': ([MuteInterval(1, 2), MuteInterval(1.5, 2.1)], [.5, 1.35, 2.6]),
        'zero': ([MuteInterval(0, 1)], [1.5, .55, 2.6]),
        'eof': ([MuteInterval(3, 4.35)], [.5, 3.5, 2.6]),
    }
    windows, points = cases[request['case']]
    asyncio.run(scope['_filter_audio_continuously']('ffmpeg', source,
        windows, 2, '512k', output, 90))
    original, rendered = probe(source), probe(output)
    assert len(rendered) == len(original) == 2
    report = []
    for track in range(2):
        assert rendered[track]['channels'] == original[track]['channels'] == channels
        assert rendered[track]['sample_rate'] == original[track]['sample_rate'] == '48000'
        assert rendered[track]['channel_layout'] == original[track]['channel_layout']
        # No track-specific timestamp reset: preserve delayed commentary.
        delta = first_packet(output, track) - first_packet(source, track)
        assert abs(delta) <= .05, (track, delta)
        rms = []
        for start in points:
            raw = execute(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(output),
                '-map', f'0:a:{track}', '-t', '0.2', '-f', 'f32le', '-c:a', 'pcm_f32le', 'pipe:1'])
            samples = np.frombuffer(raw, dtype='<f4')
            assert len(samples) > 0
            rms.append(float(np.sqrt(np.mean(samples*samples))))
        assert rms[0] > .05 and rms[2] > .05, rms
        # Interior excludes AAC boundary ringing. -80 dBFS is far below
        # the fixture's ~-17 dBFS tone and a defensible decoded silence bound.
        assert rms[1] < 0.0001, rms
        report.append({'track': track, 'channels': channels, 'rms': rms, 'pts_delta': delta})
    print(json.dumps(report))
'''


@pytest.mark.parametrize("container", CONTAINERS or ["unconfigured"])
@pytest.mark.parametrize("channels", [2, 8])
@pytest.mark.parametrize("case", ["middle", "zero", "eof"])
def test_continuous_mute_energy_all_tracks_and_delays(container, channels, case):
    request = {"volume": inspect.getsource(build_volume_filter),
               "filter": inspect.getsource(_filter_audio_continuously), "channels": channels, "case": case}
    result = subprocess.run(["docker", "exec", "-i", container, "python3", "-c", DIAGNOSTIC],
        input=json.dumps(request), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(report) == 2
