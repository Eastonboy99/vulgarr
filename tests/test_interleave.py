"""Packet-layout publication guards, lossless staging and process cleanup."""
import asyncio
import json
import sys
from pathlib import Path

import psutil
import pytest

from app.mux import finalize, interleave, remux
from app.mux.remux import RemuxError

STREAMS = [{'index': 0, 'codec_type': 'video'},
           {'index': 1, 'codec_type': 'audio'},
           {'index': 2, 'codec_type': 'audio'},
           {'index': 3, 'codec_type': 'subtitle'}]


def packet(index, pts, **kwargs):
    return dict(stream_index=index, pts_time=str(pts), **kwargs)


def test_normal_reordered_video_and_sparse_subtitles():
    interleave.check_packet_order([
        packet(0, 0, dts_time='N/A'), packet(0, .167),
        packet(1, .020), packet(2, .040), packet(0, .042),
        packet(3, 600), packet(1, .5), packet(0, .083),
    ], STREAMS)


def test_rejects_reproduced_42_second_audio_lead():
    with pytest.raises(RemuxError, match='42.348s'):
        interleave.check_packet_order([
            packet(0, 0), packet(0, .167), packet(1, 42.515),
        ], STREAMS)


def test_checks_audio_seen_before_first_video():
    with pytest.raises(RemuxError, match='Unsafe packet'):
        interleave.check_packet_order([packet(1, 42), packet(0, 0)], STREAMS)


@pytest.mark.parametrize('value', [None, 'N/A', 'nan', 'inf'])
def test_invalid_av_timestamp_rejected(value):
    with pytest.raises(RemuxError, match='usable timestamp'):
        interleave.check_packet_order([packet(0, value)], STREAMS)


def test_dts_fallback_and_unknown_sparse_timestamp():
    interleave.check_packet_order([
        {'stream_index': 0, 'dts_time': '0'},
        packet(1, .02), {'stream_index': 3},
    ], STREAMS)


def test_bounded_initial_middle_and_tail_samples(monkeypatch, tmp_path):
    calls = []
    async def probe(*args):
        return dict(streams=STREAMS, format=dict(duration='120'))
    async def execute(cmd, **kwargs):
        calls.append(cmd)
        return 0, json.dumps(dict(packets=[packet(0, 0), packet(1, .02)])), ''
    monkeypatch.setattr(interleave, 'probe', probe)
    monkeypatch.setattr(interleave, '_run', execute)
    asyncio.run(interleave.verify_packet_interleaving('probe', tmp_path/'file.mkv'))
    assert [c[c.index('-read_intervals')+1] for c in calls] == [
        '0.000000%+#3000', '60.000000%+#3000', '105.000000%+#3000']


@pytest.mark.parametrize('reply', ['{}', 'bad json', '{"packets": []}'])
def test_invalid_probe_fails_closed(monkeypatch, tmp_path, reply):
    async def probe(*args):
        return dict(streams=STREAMS)
    async def execute(*args, **kwargs):
        return 0, reply, ''
    monkeypatch.setattr(interleave, 'probe', probe)
    monkeypatch.setattr(interleave, '_run', execute)
    with pytest.raises(RemuxError, match='no usable packet'):
        asyncio.run(interleave.verify_packet_interleaving('probe', tmp_path/'file.mkv'))


@pytest.mark.parametrize('failure', [None, 'av', 'sparse', 'verify', 'cancel'])
def test_two_pass_mux_cleanup_and_preservation(monkeypatch, tmp_path, failure):
    commands, checked = [], []
    original = tmp_path/'original.mkv'
    original.write_bytes(b'untouched source')
    output = tmp_path/'staged.mkv'
    async def execute(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs['max_rss_bytes'] == finalize.INTERLEAVE_RSS_BUDGET
        Path(cmd[-1]).write_bytes(b'staged copy')
        if failure == 'cancel':
            raise asyncio.CancelledError()
        if failure == ('av' if len(commands) == 1 else 'sparse'):
            return 1, '', 'intentional failure'
        return 0, '', ''
    async def verify(probe, path):
        checked.append(path)
        if failure == 'verify' and len(checked) == 2:
            raise RemuxError('Unsafe packet interleaving')
    async def source_probe(*args):
        return {'streams': STREAMS + [{'codec_type': 'attachment'}, {'codec_type': 'data'}]}
    monkeypatch.setattr(finalize, '_run', execute)
    monkeypatch.setattr(finalize, 'probe', source_probe)
    monkeypatch.setattr(finalize, 'verify_packet_interleaving', verify)
    operation = finalize.mux_interleaved_output(
        ffmpeg_bin='ffmpeg', ffprobe_bin='probe', video_path=tmp_path/'video.mkv',
        audio_path=tmp_path/'clean.mka', metadata_path=original,
        audio_count=2, out_path=output, timeout=30)
    if failure:
        with pytest.raises(asyncio.CancelledError if failure == 'cancel' else RemuxError):
            asyncio.run(operation)
        assert not output.exists()
    else:
        asyncio.run(operation)
        assert output.read_bytes() == b'staged copy'
        assert len(checked) == 2
        dense, final = commands
        assert dense[dense.index('-max_interleave_delta')+1] == '0'
        assert '1:s?' not in dense and '1:a:2' not in dense
        assert final[final.index('-max_interleave_delta')+1] == '1000000'
        assert all(s in final for s in ['1:s?', '1:d?', '1:t?', '0:a:0', '0:a:1'])
        assert final[final.index('-map_chapters')+1] == '1'
        assert '-copyts' in final
        for spec in ['s', 'd', 't']:
            assert final[final.index(f'-map_metadata:s:{spec}:0')+1] == f'1:s:{spec}:0'
        assert all(c[c.index('-c')+1] == 'copy' for c in commands)
    assert original.read_bytes() == b'untouched source'
    assert not list(tmp_path.glob('.*.spf-blur-tmp.interleave-*'))


def test_memory_budget_kills_and_reaps_child(monkeypatch):
    children = []
    create = asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        proc = await create(*args, **kwargs)
        children.append(proc)
        return proc
    monkeypatch.setattr(remux.asyncio, 'create_subprocess_exec', capture)
    with pytest.raises(RemuxError, match='memory budget'):
        asyncio.run(remux._run([sys.executable, '-c', 'import time; time.sleep(30)'],
                               timeout=5, max_rss_bytes=1))
    assert children[0].returncode is not None
    assert not psutil.pid_exists(children[0].pid)


@pytest.mark.parametrize('cancel', [False, True])
def test_timeout_and_cancellation_reap_monitored_child(monkeypatch, cancel):
    children = []
    create = asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        proc = await create(*args, **kwargs)
        children.append(proc)
        return proc
    monkeypatch.setattr(remux.asyncio, 'create_subprocess_exec', capture)
    async def operation():
        task = asyncio.create_task(remux._run(
            [sys.executable, '-c', 'import time; time.sleep(30)'],
            timeout=5 if cancel else .05, max_rss_bytes=1024**3))
        if cancel:
            await asyncio.sleep(.05)
            task.cancel()
        await task
    with pytest.raises(asyncio.CancelledError if cancel else RemuxError):
        asyncio.run(operation())
    assert children[0].returncode is not None
    assert not psutil.pid_exists(children[0].pid)
