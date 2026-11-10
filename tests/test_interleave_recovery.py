"""Crash recovery and failure classification for interleave-safe publication."""
import asyncio
from pathlib import Path

import pytest

from app.audio.mute import MuteInterval
from app.mux import finalize, scene_blur
from app.mux.interleave import PacketInterleaveError
from app.mux.remux import MuxResourceError, RemuxError
from app.mux.tempfiles import remove_orphaned_temp_files


def test_crashed_matroska_intermediate_swept_for_mp4_source(tmp_path):
    source = tmp_path/'Movie.mp4'
    source.write_bytes(b'original')
    dead = tmp_path/'.Movie.spf-blur-tmp.interleave-dead.mkv'
    dead.write_bytes(b'orphan')
    protected = [tmp_path/'Movie - Vulgarr Edit.mkv',
                 tmp_path/'.Other.spf-blur-tmp.interleave-live.mkv',
                 tmp_path/'.Movie.spf-blur-tmp.unrelated.mkv']
    for p in protected:
        p.write_bytes(b'keep')
    assert remove_orphaned_temp_files(source, min_age_seconds=0) == [dead]
    assert source.read_bytes() == b'original'
    assert all(p.exists() for p in protected)


def test_live_interleave_file_keeps_age_protection(tmp_path):
    source = tmp_path/'Movie.mkv'
    with finalize._temporary_av_path(source, tmp_path/'staged.mkv') as dense:
        assert dense.exists()
        assert remove_orphaned_temp_files(source) == []
    assert not dense.exists()


@pytest.mark.parametrize('failure', [PacketInterleaveError, MuxResourceError])
def test_mux_safety_failure_does_not_restart_whole_movie_encode(monkeypatch, tmp_path, failure):
    source = tmp_path/'Movie.mkv'
    source.write_bytes(b'original')
    calls = []
    async def probe(*args):
        return {'streams': [], 'format': {'duration': '60'}}
    async def build(**kwargs):
        calls.append(kwargs['force_whole_file_video'])
        raise failure('test safety guard')
    monkeypatch.setattr(scene_blur, 'probe', probe)
    monkeypatch.setattr(scene_blur, 'build_blurred_video', build)
    with pytest.raises(failure):
        asyncio.run(scene_blur.apply_blur(
            video_path=source, blur_intervals=[MuteInterval(1, 2)],
            mute_intervals=[], ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe',
            video_crf=23, video_preset='fast', work_dir=tmp_path/'work'))
    assert calls == [False]
    assert source.read_bytes() == b'original'


def test_finalizer_rejects_source_as_staging_output(tmp_path):
    source = tmp_path/'Movie.mkv'
    source.write_bytes(b'original')
    with pytest.raises(RemuxError, match='must not overwrite an input'):
        asyncio.run(finalize.mux_interleaved_output(
            ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe', video_path=source,
            audio_path=source, metadata_path=source, audio_count=1,
            out_path=source, timeout=1))
    assert source.read_bytes() == b'original'
