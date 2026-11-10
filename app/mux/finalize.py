"""Lossless, memory-supervised final muxing for edited video libraries."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.mux.interleave import verify_packet_interleaving
from app.mux.remux import RemuxError, _run, probe

# AV-only buffering fixes pathological packet layout, but delayed/broken tracks
# must not exhaust the server. This is an RSS watchdog, not a hard OS limit.
INTERLEAVE_RSS_BUDGET = 1024 * 1024 * 1024


@contextmanager
def _temporary_av_path(metadata_path: Path, out_path: Path):
    """Use the title's existing orphan-sweep namespace, including after SIGKILL."""
    fd, name = tempfile.mkstemp(
        prefix=f'.{metadata_path.stem}.spf-blur-tmp.interleave-',
        suffix='.mkv', dir=out_path.parent,
    )
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


async def mux_interleaved_output(
    *, ffmpeg_bin: str, ffprobe_bin: str, video_path: Path, audio_path: Path,
    metadata_path: Path, audio_count: int, out_path: Path, timeout: float,
) -> None:
    """Interleave dense AV first, then restore sparse streams with finite flushing.

    A real HEVC edit placed 156 MB of audio between video packets two and three.
    Decoder/duration checks passed, but Plex stalled. Disabling early interleave
    flushing repaired the file without re-encoding. Do that ONLY for video and
    real audio tracks: subtitles/data can have movie-length gaps. Supervise RSS
    even in this pass because missing/delayed audio can also be sparse.

    The second pass takes AV together from one already-interleaved input and
    restores original subtitles, data, attachments, chapters and global tags.
    Its finite flush threshold avoids waiting indefinitely for sparse packets.
    Check both passes rather than assume sparse-stream restoration stays safe.
    A private on-disk intermediate costs one additional full AV copy temporarily;
    it is removed after success, failure, timeout or cancellation. The caller
    owns atomic publication; this function must only receive a staging path.
    """
    if out_path.resolve() in {p.resolve() for p in (video_path, audio_path, metadata_path)}:
        raise RemuxError('Interleave staging output must not overwrite an input')
    original = await probe(ffprobe_bin, metadata_path)
    sparse_counts = {
        spec: sum(s.get('codec_type') == kind for s in original['streams'])
        for spec, kind in [('s', 'subtitle'), ('d', 'data'), ('t', 'attachment')]
    }
    with _temporary_av_path(metadata_path, out_path) as dense_path:
        cmd = [ffmpeg_bin, '-y', '-nostdin', '-loglevel', 'error',
               '-threads', '1', '-i', str(video_path)]
        if audio_count:
            cmd += ['-threads', '1', '-i', str(audio_path)]
        cmd += ['-map', '0:v:0']
        for i in range(audio_count):
            cmd += ['-map', f'1:a:{i}', f'-map_metadata:s:a:{i}', f'1:s:a:{i}']
        cmd += ['-map_metadata:s:v:0', '0:s:v:0',
                '-c', 'copy', '-map_metadata', '-1', '-map_chapters', '-1',
                '-max_interleave_delta', '0', str(dense_path)]
        try:
            code, _out, err = await _run(
                cmd, timeout=timeout, max_rss_bytes=INTERLEAVE_RSS_BUDGET
            )
            if code:
                raise RemuxError(f'ffmpeg AV interleave mux failed: {err.strip()[-4000:]}')
            await verify_packet_interleaving(ffprobe_bin, dense_path)

            cmd = [ffmpeg_bin, '-y', '-nostdin', '-loglevel', 'error', '-copyts',
                   '-threads', '1', '-i', str(dense_path),
                   '-threads', '1', '-i', str(metadata_path), '-map', '0:v:0']
            for i in range(audio_count):
                cmd += ['-map', f'0:a:{i}', f'-map_metadata:s:a:{i}', f'0:s:a:{i}']
            # Explicit metadata mapping disables FFmpeg automatic stream-tag
            # copying. Restore EVERY mapped type, including attachment MIME and
            # filename tags, rather than only audio titles/languages.
            cmd += ['-map_metadata:s:v:0', '0:s:v:0']
            for spec, count in sparse_counts.items():
                for i in range(count):
                    cmd += [f'-map_metadata:s:{spec}:{i}', f'1:s:{spec}:{i}']
            cmd += ['-map', '1:s?', '-map', '1:d?', '-map', '1:t?',
                    '-c', 'copy', '-map_metadata', '1', '-map_chapters', '1',
                    '-max_interleave_delta', '1000000', '-avoid_negative_ts', 'disabled',
                    '-metadata', 'title=Vulgarr Edit',
                    '-metadata:s:v:0', 'title=Vulgarr Edit', str(out_path)]
            code, _out, err = await _run(
                cmd, timeout=timeout, max_rss_bytes=INTERLEAVE_RSS_BUDGET
            )
            if code:
                raise RemuxError(f'ffmpeg final mux failed: {err.strip()[-4000:]}')
            await verify_packet_interleaving(ffprobe_bin, out_path)
        except BaseException:
            out_path.unlink(missing_ok=True)
            raise
