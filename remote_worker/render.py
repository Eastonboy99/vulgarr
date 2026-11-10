"""Whole-video recovery rendering on the worker, without codec splices.

Segment offload alone cannot cover jobs whose split/keyframe planning fails
before the first segment encode. This API renders one continuous video stream,
then lets vulgarr reuse its existing audio, metadata, verification and atomic
publication paths. Pollable jobs avoid hours-long idle HTTP requests; source
URLs are restricted capabilities and must never appear in status/error text.
"""
import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()
encode_slot = asyncio.Semaphore(1)
_jobs: dict = {}
_completed = 0
_latency = 0.0
_RENDER_TIMEOUT = 24 * 60 * 60
_RESULT_TTL = 3600
_MAX_JOBS = 2
# NVDEC downloads NV12/P010 frames. boxblur on the deployed FFmpeg 4.4 build
# corrupts the right half of their chroma while enabled unless converted to
# the source's canonical planar format first. Advertise this fix to clients.
FILTER_LAYOUT = 'planar-v1'


class RenderRequest(BaseModel):
    source_url: str
    source_codec: str = Field(max_length=40)
    duration: float = Field(gt=0, le=_RENDER_TIMEOUT, allow_inf_nan=False)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    pix_fmt: str = Field(pattern=r'^[a-zA-Z0-9_]+$', max_length=64)
    crf: int = Field(ge=0, le=51)
    preset: str = Field(pattern=r'^[a-zA-Z0-9_-]+$', max_length=40)
    filter_complex: str = Field(max_length=100000)


def validate_source(url: str) -> None:
    trusted = urlsplit(os.getenv('SPF_SOURCE_BASE_URL', ''))
    supplied = urlsplit(url)
    if (trusted.scheme not in {'http', 'https'} or supplied.scheme != trusted.scheme
            or supplied.netloc != trusted.netloc or supplied.path != '/internal/source'):
        raise ValueError('Source origin is not configured or permitted')


def validate_blur_filter(graph: str) -> None:
    # Same restricted grammar as /encode, not an arbitrary ffmpeg filter API.
    import re
    label = r'(?:0:v|vb(?:_stage\d+)?)'
    number = r'\d+\.\d{3}'
    term = rf'between\(t,{number},{number}\)'
    stage = (rf'\[{label}\]boxblur=luma_radius=\d+:luma_power=\d+:'
             rf"chroma_radius=\d+:chroma_power=\d+:enable='{term}(?:\+{term})*'\[{label}\]")
    if not re.fullmatch(rf'(?:{stage})(?:;{stage})*|\[0:v\]null\[vb\]', graph):
        raise ValueError('Unsupported blur graph')


def software_codec(source: str) -> str:
    """Match the host's _reencode_video_codec, including its H.264 default."""
    return {'hevc': 'libx265', 'h265': 'libx265', 'vp9': 'libvpx-vp9',
            'av1': 'libsvtav1', 'mpeg2video': 'mpeg2video', 'mpeg4': 'mpeg4'}.get(source, 'libx264')


def _command(params: RenderRequest, output: Path, backend: str, cuda_decode: bool) -> list[str]:
    cmd = ['ffmpeg', '-y', '-nostdin', '-loglevel', 'error', '-progress', 'pipe:1']
    if cuda_decode:
        cmd += ['-hwaccel', 'cuda']
    # Do not let boxblur operate directly on NVDEC's semiplanar NV12/P010
    # download. Explicitly convert to the ffprobe source format (e.g.
    # yuv420p/yuv420p10le); this preserves chroma sampling and bit depth, unlike
    # forcing every movie through 8-bit yuv420p. Keep the conversion outside
    # the enable window so switching blur on/off never changes frame layout.
    # The incoming graph was validated before submission; the intermediate
    # label cannot collide with its restricted vb/vb_stageN labels.
    graph = (f'[0:v]format=pix_fmts={params.pix_fmt}[vulgarr_planar];'
             + params.filter_complex.replace('[0:v]', '[vulgarr_planar]'))
    cmd += ['-rw_timeout', '30000000', '-i', params.source_url,
            '-filter_complex', graph, '-map', '[vb]']
    if backend == 'NVENC':
        codec = 'hevc_nvenc' if params.source_codec in {'hevc', 'h265'} else 'h264_nvenc'
        cmd += ['-c:v', codec, '-preset', 'p4', '-rc', 'vbr', '-cq', str(params.crf), '-b:v', '0']
    else:
        codec = software_codec(params.source_codec)
        cmd += ['-c:v', codec]
        if codec in {'libx264', 'libx265'}:
            cmd += ['-crf', str(params.crf), '-preset', params.preset]
        elif codec == 'libvpx-vp9':
            cmd += ['-crf', str(params.crf), '-b:v', '0', '-cpu-used', '4']
        elif codec == 'libsvtav1':
            cmd += ['-crf', str(params.crf), '-preset', '8']
        else:
            # Native MPEG encoders do not understand CRF or x26x presets.
            cmd += ['-q:v', str(max(2, min(31, round(params.crf * 31 / 51))))]
    # Preserve bit depth/chroma where possible; post-encode probing rejects an
    # encoder's silent fallback to a different decoded pixel format.
    cmd += ['-pix_fmt', params.pix_fmt, '-an', '-sn', '-f', 'matroska', str(output)]
    return cmd


async def _run_ffmpeg(command: list[str], job: dict) -> int:
    proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                              stderr=asyncio.subprocess.DEVNULL)
    async def progress():
        async for line in proc.stdout:
            if line.startswith(b'out_time_us='):
                try:
                    seconds = int(line.split(b'=', 1)[1]) / 1_000_000
                    job['progress'] = max(0.0, min(0.99, seconds / job['params'].duration))
                except ValueError:
                    pass
    try:
        await asyncio.wait_for(asyncio.gather(progress(), proc.wait()), timeout=_RENDER_TIMEOUT)
        return proc.returncode
    finally:
        # Poll-client cancellation and worker shutdown must stop the actual
        # ffmpeg, not merely abandon its coroutine with a live GPU process.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def _valid_output(path: Path, params: RenderRequest) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    proc = await asyncio.create_subprocess_exec(
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        data, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode:
            return False
        result = json.loads(data)
        streams = result['streams']
        expected_codec = {'libx265': 'hevc', 'libx264': 'h264', 'libvpx-vp9': 'vp9',
                          'libsvtav1': 'av1'}.get(software_codec(params.source_codec), params.source_codec)
        video = streams[0]
        return (len(streams) == 1 and video['codec_type'] == 'video'
                and video['codec_name'] == expected_codec
                and video['width'] == params.width and video['height'] == params.height
                and video['pix_fmt'] == params.pix_fmt
                and abs(float(result['format']['duration']) - params.duration) <= 2.0)
    except Exception:
        return False
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


async def _perform_render(job: dict, started: float):
    global _completed, _latency
    async with encode_slot:
        job['state'] = 'running'
        params = job['params']
        attempts = [('NVENC', True), ('NVENC', False)] if params.source_codec in {'h264', 'hevc', 'h265'} else []
        attempts.append(('worker CPU', False))
        for backend, cuda in attempts:
            job.update(backend=backend, progress=0.0)
            job['output'].unlink(missing_ok=True)
            code = await _run_ffmpeg(_command(params, job['output'], backend, cuda), job)
            if code == 0 and await _valid_output(job['output'], params):
                job.update(state='done', progress=1.0)
                _completed += 1
                _latency += time.monotonic() - started
                return
        job.update(state='failed', error='Worker GPU and software rendering failed')


async def _render(job: dict):
    started = time.monotonic()
    try:
        # One deadline covers queueing AND all hardware/software attempts.
        # A vanished client cannot leave three successive day-long encodes.
        await asyncio.wait_for(_perform_render(job, started), timeout=_RENDER_TIMEOUT)
    except asyncio.CancelledError:
        job['state'] = 'cancelled'
        raise
    except Exception:
        job.update(state='failed', error='Worker rendering failed')
    finally:
        job['finished'] = time.monotonic()
        # A client can begin downloading as soon as _perform_render marks
        # done, before this wrapper's wait_for resumes. Do not erase that
        # output merely because its state already became downloading.
        if job['state'] in {'failed', 'cancelled'}:
            job['directory'].cleanup()


def _discard(identity: str):
    job = _jobs.pop(identity, None)
    if job:
        job['directory'].cleanup()


def render_stats() -> dict:
    return {'render_requests_served': _completed, 'render_latency_seconds': _latency,
            'render_in_flight': sum(j['state'] == 'running' for j in _jobs.values()),
            'render_queued': sum(j['state'] == 'queued' for j in _jobs.values()),
            'whole_video_render': True, 'render_filter_layout': FILTER_LAYOUT}


def _get(identity: str) -> dict:
    job = _jobs.get(identity)
    if job is None:
        raise HTTPException(404, 'Render job not found')
    return job


@router.post('/render', status_code=202)
async def start_render(params: RenderRequest):
    try:
        validate_source(params.source_url)
        validate_blur_filter(params.filter_complex)
    except ValueError:
        raise HTTPException(422, 'Invalid render source or blur graph')
    now = time.monotonic()
    for identity, job in list(_jobs.items()):
        ttl = _RENDER_TIMEOUT if job['state'] == 'downloading' else _RESULT_TTL
        if job['state'] in {'done', 'failed', 'cancelled', 'downloading'} and now - job.get('finished', now) > ttl:
            _discard(identity)
    if len(_jobs) >= _MAX_JOBS:
        raise HTTPException(429, 'Worker render queue is full')
    root = os.getenv('SPF_RENDER_WORK_DIR')
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix='vulgarr-render-', dir=root)
    identity = uuid.uuid4().hex
    job = {'state': 'queued', 'backend': 'pending', 'progress': 0.0, 'error': None,
           'params': params, 'directory': directory, 'output': Path(directory.name)/'video.mkv'}
    _jobs[identity] = job
    job['task'] = asyncio.create_task(_render(job))
    return {'id': identity, 'filter_layout': FILTER_LAYOUT}


@router.get('/render/{identity}')
async def render_status(identity: str):
    job = _get(identity)
    return {key: job[key] for key in ('state', 'backend', 'progress', 'error')}


class RenderFileResponse(FileResponse):
    """Release on send errors too, not just a successful background callback.

    Keep the directory reference so a concurrent cancellation's failed unlink
    (notably an open file on Windows) can be retried after FileResponse closes
    its handle. Registry mutations stay on the event loop, not a pool thread.
    """
    def __init__(self, identity, job):
        super().__init__(job['output'], media_type='video/x-matroska')
        self.identity = identity
        self.directory = job['directory']

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            _jobs.pop(self.identity, None)
            self.directory.cleanup()


@router.get('/render/{identity}/output')
async def render_output(identity: str):
    job = _get(identity)
    if job['state'] != 'done':
        raise HTTPException(409, 'Render is not complete')
    job['state'] = 'downloading'
    job['finished'] = time.monotonic()
    return RenderFileResponse(identity, job)


@router.delete('/render/{identity}')
async def cancel_render(identity: str):
    job = _get(identity)
    job['task'].cancel()
    await asyncio.gather(job['task'], return_exceptions=True)
    _discard(identity)
    return {'ok': True}


async def startup_renders():
    """Reclaim hard-killed jobs only in an explicitly dedicated worker volume.

    A shared system /tmp may belong to other worker processes, so never sweep
    it. Each process must have its own SPF_RENDER_WORK_DIR (as compose does).
    """
    root = os.getenv('SPF_RENDER_WORK_DIR')
    if root:
        for path in Path(root).glob('vulgarr-render-*'):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)


async def shutdown_renders():
    for job in _jobs.values():
        job['task'].cancel()
    await asyncio.gather(*(j['task'] for j in _jobs.values()), return_exceptions=True)
    for identity in list(_jobs):
        _discard(identity)
