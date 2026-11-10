"""Opt-in real FFmpeg/NVENC qualification without changing worker daemons.

SPF_TEST_RENDER_CONTAINERS=worker-0,worker-1 python -m pytest tests/test_interleave_media.py -q
Candidate modules run in an isolated Python process; all media is synthetic and
removed afterward. The worker lacks the server-only psutil dependency, so this
harness provides its RSS interface using Linux procfs. Host tests exercise real
psutil supervision and process cleanup separately.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

CONTAINERS = [s.strip() for s in os.getenv('SPF_TEST_RENDER_CONTAINERS', '').split(',') if s.strip()]
pytestmark = pytest.mark.skipif(not CONTAINERS, reason='Opt-in real Docker media qualification')

DIAGNOSTIC = r'''
import ast, asyncio, json, subprocess, sys, tempfile, types
from pathlib import Path
request = json.load(sys.stdin)
for name in ['app', 'app.mux']:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
psutil = types.ModuleType('psutil')
class Error(Exception): pass
class NoSuchProcess(Error): pass
class Process:
    def __init__(self, pid): self.pid = pid
    def memory_info(self):
        try:
            lines = Path(f'/proc/{self.pid}/status').read_text().splitlines()
            rss = next(int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:'))
        except (FileNotFoundError, StopIteration):
            raise NoSuchProcess(self.pid)
        return types.SimpleNamespace(rss=rss)
psutil.Error, psutil.NoSuchProcess, psutil.Process = Error, NoSuchProcess, Process
sys.modules['psutil'] = psutil
# Only the existing subprocess/probe boundary is needed; avoid loading the
# server ORM, authentication and classifier dependencies into a GPU worker.
tree = ast.parse(request['remux'])
selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in ['RemuxError', 'MuxResourceError', '_watch_rss', '_run', 'probe']]
module = types.ModuleType('app.mux.remux')
module.__dict__.update(asyncio=asyncio, json=json, psutil=psutil, Path=Path)
sys.modules[module.__name__] = module
exec(compile(ast.Module(body=selected, type_ignores=[]), '<candidate-remux>', 'exec'), module.__dict__)
for name in ['interleave', 'finalize']:
    module = types.ModuleType('app.mux.'+name)
    sys.modules[module.__name__] = module
    exec(compile(request[name], '<candidate-'+name+'>', 'exec'), module.__dict__)
finalize = sys.modules['app.mux.finalize']

def execute(args):
    p = subprocess.run(args, capture_output=True, timeout=90)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return p.stdout

def probe(path):
    return json.loads(execute(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
        '-show_chapters', '-show_data_hash', 'sha256', '-of', 'json', str(path)]))

def vint(data, pos, identifier=False):
    width = 1
    while not data[pos] & (0x80 >> (width-1)):
        width += 1
    value = int.from_bytes(data[pos:pos+width], 'big')
    return (value if identifier else value & ((1 << (7*width))-1)), width

def elements(data, start=0, end=None):
    end = len(data) if end is None else end
    while start < end:
        identifier, n = vint(data, start, True)
        size, k = vint(data, start+n)
        body = start+n+k
        stop = min(end, body+size)
        yield identifier, start, body, stop
        start = stop

def element(identifier, payload):
    width = next(n for n in range(1,9) if len(payload) < (1 << (7*n))-1)
    return (identifier.to_bytes((identifier.bit_length()+7)//8, 'big') +
            ((1 << (7*width)) | len(payload)).to_bytes(width, 'big') + payload)

def frontload_audio(source, target):
    # Deterministic malformed-layout fixture: preserve every compressed block,
    # PTS and metadata, but place all audio/subtitle blocks between video packets
    # two and three. This reproduces the real 156-MB-gap failure without a huge
    # movie. Rebuild cluster timestamps and omit obsolete seek/cue offsets.
    data = source.read_bytes()
    segment = next(e for e in elements(data) if e[0] == 0x18538067)
    children = list(elements(data, segment[2], segment[3]))
    blocks, preserved = [], []
    for identifier, begin, body, end in children:
        if identifier != 0x1F43B675:
            if identifier not in [0x114D9B74, 0x1C53BB6B, 0xBF]:
                preserved.append(data[begin:end])
            continue
        cluster = list(elements(data, body, end))
        base = next(int.from_bytes(data[b:e], 'big') for i,a,b,e in cluster if i == 0xE7)
        for i,a,b,e in cluster:
            if i not in [0xA3, 0xA0]:
                continue
            nested = list(elements(data,b,e)) if i == 0xA0 else [(i,a,b,e)]
            block = next(z for z in nested if z[0] in [0xA1, 0xA3])
            _, start, payload, stop = block
            track, track_size = vint(data, payload)
            pts = base + int.from_bytes(data[payload+track_size:payload+track_size+2], 'big', signed=True)
            timestamp = max(0, pts)
            relative = pts-timestamp
            content = (data[payload:payload+track_size] + relative.to_bytes(2,'big',signed=True)
                       + data[payload+track_size+2:stop])
            rewritten = element(block[0], content)
            if i == 0xA0:
                rewritten = element(i, b''.join(rewritten if q[0] == 0xA1 else data[q[1]:q[3]] for q in nested))
            cluster_data = element(0xE7, timestamp.to_bytes(max(1,(timestamp.bit_length()+7)//8),'big')) + rewritten
            blocks.append((track, element(0x1F43B675, cluster_data)))
    video = [b for t,b in blocks if t == 1]
    other = [b for t,b in blocks if t != 1]
    payload = b''.join(preserved + video[:2] + other + video[2:])
    target.write_bytes(data[:segment[1]] + element(0x18538067, payload))


def video_hashes(path):
    data = execute(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
                    '-map', '0:v:0', '-an', '-f', 'framemd5', '-']).decode()
    return [line.split(',')[-1].strip() for line in data.splitlines() if not line.startswith('#')]

with tempfile.TemporaryDirectory(prefix='vulgarr-interleave-test-') as folder:
    root = Path(folder)
    video, source, output = [root/name for name in ['video.mkv', 'original.mkv', 'edited.mkv']]
    codec, fmt = request['codec'], request['fmt']
    encode = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=320x180:rate=24000/1001', '-t', '24', '-vf', 'format='+fmt,
        '-c:v', codec+'_nvenc', '-preset', 'fast', '-bf', '3',
        '-avoid_negative_ts', 'disabled', str(video)]
    try:
        execute(encode)
    except RuntimeError as exc:
        # Pascal supports HEVC but not HEVC B-frames; Turing exercises reorder.
        if codec != 'hevc' or 'No capable devices found' not in str(exc):
            raise
        video.unlink(missing_ok=True)
        encode[encode.index('-bf')+1] = '0'
        execute(encode)
    subtitles = root/'sparse.srt'
    subtitles.write_text('1\n00:00:18,000 --> 00:00:20,000\nSparse subtitle\n')
    metadata = root/'metadata.txt'
    metadata.write_text(';FFMETADATA1\ncomment=Preserve global metadata\n'
                        '[CHAPTER]\nTIMEBASE=1/1000\nSTART=1000\nEND=2000\ntitle=Chapter one\n')
    attachment = root/'test.bin'
    attachment.write_bytes(b'Preserve attachment payload\x00\x01')
    main_audio_delay = '0.020' if codec == 'h264' else '2.500'
    execute(['ffmpeg', '-v', 'error', '-i', str(video),
        '-itsoffset', main_audio_delay, '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=7.1',
        '-itsoffset', '0.040', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
        '-i', str(subtitles), '-f', 'ffmetadata', '-i', str(metadata),
        '-t', '24', '-map', '0:v:0', '-map', '1:a:0', '-map', '2:a:0', '-map', '3:s:0',
        '-map_metadata', '4', '-map_chapters', '4', '-c:v', 'copy', '-c:a', 'aac',
        '-ac:a:1', '2', '-b:a', '128k', '-c:s', 'srt',
        '-metadata:s:a:0', 'title=Clean 7.1', '-metadata:s:a:0', 'language=eng',
        '-metadata:s:a:1', 'title=Commentary', '-metadata:s:a:1', 'language=deu',
        '-disposition:a:0', 'default', '-disposition:a:1', '0',
        '-attach', str(attachment), '-metadata:s:t:0', 'mimetype=application/octet-stream',
        '-avoid_negative_ts', 'disabled', str(source)])
    broken = root/'frontloaded.mkv'
    frontload_audio(source, broken)
    guard = sys.modules['app.mux.interleave']
    try:
        asyncio.run(guard.verify_packet_interleaving('ffprobe', broken))
    except guard.PacketInterleaveError as exc:
        assert 'leads video' in str(exc), str(exc)
    else:
        raise AssertionError('Fixture must reproduce unsafe AV packet ordering')
    asyncio.run(finalize.mux_interleaved_output(ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe',
        video_path=broken, audio_path=broken, metadata_path=broken, audio_count=2,
        out_path=output, timeout=90))
    before, after = probe(source), probe(output)
    assert len(before['streams']) == len(after['streams']) == 5
    assert after['streams'][0]['codec_name'] == codec
    assert after['streams'][0]['pix_fmt'] == fmt
    before_audio = [s for s in before['streams'] if s['codec_type']=='audio']
    after_audio = [s for s in after['streams'] if s['codec_type']=='audio']
    for a, b in zip(before_audio, after_audio):
        for key in ['codec_name', 'profile', 'sample_rate', 'channels', 'channel_layout']:
            assert a.get(key) == b.get(key), (key, a.get(key), b.get(key))
        for key in ['title', 'language']:
            assert a['tags'].get(key) == b['tags'].get(key), (key, a['tags'], b['tags'])
        assert a['disposition']['default'] == b['disposition']['default']
        assert abs(float(a['start_time'])-float(b['start_time'])) < .003
    for kind in ['subtitle', 'attachment']:
        a = next(s for s in before['streams'] if s['codec_type']==kind)
        b = next(s for s in after['streams'] if s['codec_type']==kind)
        assert a.get('extradata_hash') == b.get('extradata_hash')
        if kind == 'attachment':
            assert a['extradata_hash'] == b['extradata_hash']
        assert a.get('codec_name') == b.get('codec_name')
    def subtitle_text(path):
        return execute(['ffmpeg', '-v', 'error', '-copyts', '-i', str(path),
                        '-map', '0:s:0', '-c:s', 'srt', '-f', 'srt', 'pipe:1'])
    assert subtitle_text(source) == subtitle_text(output)
    def audio_hashes(path, index):
        return json.loads(execute(['ffprobe', '-v', 'error', '-select_streams', f'a:{index}',
            '-read_intervals', '%+#20', '-show_packets', '-show_data_hash', 'sha256',
            '-show_entries', 'packet=data_hash', '-of', 'json', str(path)]))['packets']
    for i in [0, 1]:
        assert audio_hashes(source, i) == audio_hashes(output, i), 'Changed compressed audio'
    assert before['chapters'] == after['chapters'], (before['chapters'], after['chapters'])
    assert after['format']['tags']['COMMENT'] == 'Preserve global metadata'
    assert after['format']['tags']['title'] == 'Vulgarr Edit'
    hashes = video_hashes(video)
    assert len(hashes) >= 575, len(hashes)
    assert hashes == video_hashes(output), 'Stream-copy changed decoded frames'
    assert not list(root.glob('.*.spf-blur-tmp.interleave-*'))
    print(json.dumps(dict(codec=codec, pix_fmt=fmt, frames=len(hashes), streams=5,
                         sparse_subtitles=True, attachments=True, chapters=True, audio_delays=True)))
'''


@pytest.mark.parametrize('container', CONTAINERS or ['unconfigured'])
@pytest.mark.parametrize('codec,fmt', [('h264', 'yuv420p'), ('hevc', 'yuv420p'), ('hevc', 'yuv420p10le')])
def test_real_two_pass_mux_preserves_media(container, codec, fmt):
    root = Path(__file__).resolve().parents[1]
    payload = {name: (root/'app'/'mux'/f'{name}.py').read_text()
               for name in ['remux', 'interleave', 'finalize']}
    payload.update(codec=codec, fmt=fmt)
    result = subprocess.run(['docker', 'exec', '-i', container, 'python3', '-c', DIAGNOSTIC],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stderr[-5000:]
    report = json.loads(result.stdout)
    assert report['frames'] >= 575 and report['streams'] == 5
