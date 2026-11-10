"""Bounded packet-order checks for stream-copy publication.

Duration and decoder checks cannot detect audio placed tens of megabytes before
its corresponding video. Probe a bounded number of packets at the beginning,
middle and end instead. This is a sampled safeguard, not a full-file proof.
"""

import json
import math
from pathlib import Path

from app.mux.remux import RemuxError, _run, probe

class PacketInterleaveError(RemuxError):
    """A publication-layout failure, not a reason to re-encode the video."""


# Normal B-frame reordering and Matroska clustering are comfortably below this.
MAX_AUDIO_LEAD_SECONDS = 5.0
PACKETS_PER_SAMPLE = 3000


def check_packet_order(packets: list[dict], streams: list[dict]) -> None:
    """Reject excessive audio lead in demux order, ignoring sparse stream types.

    Compare presentation timestamps, not byte distances (bitrate varies greatly).
    B-frame PTS can move backwards, so retain the greatest video PTS seen. A
    missing DTS on initial reordered video packets is normal; missing *both*
    timestamps is not. Never shift timestamps to disguise an ordering defect.
    """
    kinds = {s.get('index', i): s.get('codec_type') for i, s in enumerate(streams)}
    video = next((i for i, kind in kinds.items() if kind == 'video'), None)
    latest_video = None
    pending_audio = {}
    for packet in packets:
        index = packet.get('stream_index')
        if index != video and kinds.get(index) != 'audio':
            continue
        timestamp = None
        for field in ('pts_time', 'dts_time'):
            try:
                value = float(packet.get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                timestamp = value
                break
        if timestamp is None:
            raise PacketInterleaveError('Interleave verification found an AV packet without a usable timestamp')
        if index == video:
            latest_video = timestamp if latest_video is None else max(latest_video, timestamp)
        else:
            pending_audio[index] = max(pending_audio.get(index, timestamp), timestamp)
        if latest_video is not None:
            for audio_index, audio_pts in pending_audio.items():
                lead = audio_pts - latest_video
                if lead > MAX_AUDIO_LEAD_SECONDS:
                    raise PacketInterleaveError(
                        f'Unsafe packet interleaving: audio stream {audio_index} leads '
                        f'video by {lead:.3f}s (limit {MAX_AUDIO_LEAD_SECONDS:.1f}s)'
                    )


async def verify_packet_interleaving(ffprobe_bin: str, path: Path) -> None:
    """Sample AV packet layout with finite probe work and fail closed on errors.

    Sparse subtitles/data/attachments never participate in the lead calculation.
    Each sample is capped at 3,000 packets and a 60-second timeout.
    Video-only worker renders and audio-only masters do not need this check.
    """
    metadata = await probe(ffprobe_bin, path)
    streams = metadata.get('streams', [])
    if not any(s.get('codec_type') == 'video' for s in streams) or not any(
        s.get('codec_type') == 'audio' for s in streams
    ):
        return
    duration = float(metadata.get('format', {}).get('duration', 0) or 0)
    starts = sorted({0.0, max(0.0, duration / 2), max(0.0, duration - 15)})
    for start in starts:
        code, stdout, stderr = await _run([
            ffprobe_bin, '-v', 'error', '-read_intervals', f'{start:.6f}%+#{PACKETS_PER_SAMPLE}',
            '-show_entries', 'packet=stream_index,pts_time,dts_time', '-of', 'json', str(path),
        ], timeout=60)
        if code:
            raise PacketInterleaveError(f'Packet interleave probe failed: {stderr[-1500:]}')
        try:
            packets = json.loads(stdout)['packets']
            if not isinstance(packets, list) or not packets:
                raise ValueError('empty packet sample')
        except (ValueError, KeyError, TypeError) as exc:
            raise PacketInterleaveError('Packet interleave probe returned no usable packet sample') from exc
        if start == 0:
            video = next(s.get('index', i) for i, s in enumerate(streams)
                         if s.get('codec_type') == 'video')
            if not any(p.get('stream_index') == video for p in packets):
                raise PacketInterleaveError('Initial packet sample contains no video; cannot verify AV interleaving')
        check_packet_order(packets, streams)
