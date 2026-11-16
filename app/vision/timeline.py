"""Measure the visual horizon, not the longest audio/subtitle/container duration.

A timeout-bounded tail-packet probe verifies the last displayed video frame.
Duration metadata is a seek hint, never permission to omit an unverified tail.
"""
from dataclasses import dataclass
import json
import math

from app.mux.remux import _run, probe


@dataclass(frozen=True)
class VisualTimeline:
    container_duration: float
    video_end: float
    last_sample_time: float


def _seconds(value):
    try:
        if isinstance(value, str) and ':' in value:
            h,m,s = value.split(':')
            number = int(h)*3600 + int(m)*60 + float(s)
        else:
            number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _frame_duration(video):
    for key in ('avg_frame_rate', 'r_frame_rate'):
        try:
            a,b = str(video.get(key, '')).split('/')
            rate = float(a)/float(b)
            if math.isfinite(rate) and rate > 0:
                return 1/rate
        except (ValueError, ZeroDivisionError):
            pass
    return None


async def probe_visual_timeline(ffprobe_bin, path, info=None):
    info = info if info is not None else await probe(ffprobe_bin, path)
    container = _seconds(info.get('format', {}).get('duration'))
    origin = _seconds(info.get('format', {}).get('start_time', 0))
    videos = [s for s in info.get('streams', []) if s.get('codec_type') == 'video'
              and not s.get('disposition', {}).get('attached_pic')]
    if container is None or container <= 0 or origin is None or not videos:
        raise ValueError('Cannot establish a finite visual timeline; previous edit retained')
    video = videos[0]
    frame_duration = _frame_duration(video)
    start = _seconds(video.get('start_time', origin))
    if start is None or abs(start-origin) > .25:
        raise ValueError('Unsupported delayed video timeline; cannot assure frame coverage')
    # Probe past the stated container end, so an incorrect short track tag
    # cannot silently truncate sampling. Select the actual playback video.
    selector = str(video['index']) if type(video.get('index')) is int else 'v:0'
    hints = [container]
    declared = _seconds(video.get('duration'))
    if declared is not None and declared > 0:
        hints.append(start-origin+declared)
    for key,value in video.get('tags', {}).items():
        if key.upper() == 'DURATION':
            tag = _seconds(value)
            if tag is not None and 0 < tag < container:
                hints.append(tag)
    for hint in dict.fromkeys(hints):
        seek = max(0., hint-60)
        # Read to EOF: a fixed-length interval can end early after a long-GOP
        # backward seek, or conceal later packets behind a stale duration tag.
        # Only packet metadata is retained; no images or inference are needed.
        code,out,_err = await _run([ffprobe_bin, '-v', 'error', '-select_streams', selector,
            '-read_intervals', f'{origin+seek:.6f}%',
            '-show_entries', 'packet=pts_time,duration_time', '-of', 'json', str(path)], timeout=120)
        if code:
            raise ValueError('Video tail timing probe failed; previous edit retained')
        try:
            packets = json.loads(out)['packets']
            if not isinstance(packets, list) or any(not isinstance(p, dict) for p in packets):
                raise ValueError('Invalid packet records')
        except (ValueError, KeyError, TypeError):
            raise ValueError('Video tail probe returned invalid timing evidence') from None
        measured = []
        for packet in packets:
            pts = _seconds(packet.get('pts_time'))
            length = _seconds(packet.get('duration_time'))
            length = length if length is not None and length > 0 else frame_duration
            if pts is None or length is None or length <= 0:
                raise ValueError('Video tail packet lacks usable timing; cannot assure visual coverage')
            measured.append((pts-origin, length))
        if not measured:
            continue  # Audio-only tail: try the declared video horizon, verifying it too.
        end = max(t+d for t,d in measured)
        last_pts,last_length = max(measured)
        if end > container+max(.25,last_length*2):
            raise ValueError('Video duration conflicts with bounded tail evidence; previous edit retained')
        if end <= 0:
            continue
        # Forward seeking after the last frame's PTS can return no JPEG even
        # while it is displayed. Stay just before that PTS, but never back up
        # into an earlier frame when the final displayed frame has a long hold.
        previous = [t for t,_ in measured if t < last_pts]
        epsilon = min(.001, last_length/2, (last_pts-max(previous))/2 if previous else last_length/2)
        return VisualTimeline(container, end, max(0., last_pts-epsilon))
    raise ValueError('No video tail packets available; cannot assure visual coverage')
