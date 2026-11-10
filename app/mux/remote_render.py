"""Client for pollable worker whole-video recovery jobs.

This is deliberately separate from segment splicing: whole-video NVENC output
never has to redefine source SPS/PPS midstream. Download and validate into a
private sibling, then atomically install only the intermediate video. Existing
final mux/verification/publication still owns the user-visible edited file.
"""
import asyncio
import logging
import math
import os
import re
from urllib.parse import urlsplit

import httpx

from app.vision.remote_source import remote_source_enabled, source_url

logger = logging.getLogger(__name__)
_POLL_SECONDS = 2.0
_REQUIRED_FILTER_LAYOUT = 'planar-v1'


class IncompatibleRenderWorker(RuntimeError):
    """Controlled local diagnostic, never a worker-supplied error string."""


async def try_remote_render(*, video_path, out_path, source_codec, source_video,
                            duration, crf, preset, filter_complex, timeout,
                            validate, on_progress=None, on_stage=None) -> bool:
    """Try recovery offload without installing an unverified checkpoint.

    ``source_video`` is the original ffprobe video stream. ``validate`` must
    check the downloaded video's identity and decode windows before accepting
    it. One deadline covers submission, queueing, encoding, download and checks.
    Ordinary worker/validation failures return False for local recovery; task
    cancellation propagates after attempting to cancel the worker subprocess.
    """
    url = os.getenv('SPF_REMOTE_ENCODE_URL', '').strip().rstrip('/')
    if not url:
        return False
    if not remote_source_enabled():
        message = ('Remote whole-file blur unavailable: configure SPF_INTERNAL_BASE_URL '
                   'and SPF_INTERNAL_TOKEN; using server CPU')
        logger.warning(message)
        if on_stage:
            await on_stage(message)
        return False
    try:
        parsed = urlsplit(url)
        worker_label = f'{parsed.scheme}://{parsed.netloc.rsplit("@", 1)[-1]}'
    except ValueError:
        worker_label = 'configured worker'
    identity = None
    temporary = out_path.with_name(out_path.name + '.remote.tmp')
    maximum_progress = 0.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5)) as client:
        async def execute():
            nonlocal identity, maximum_progress
            temporary.unlink(missing_ok=True)  # Reclaim a hard-killed partial download.
            if on_stage:
                await on_stage(f'Submitting whole-file blur to remote worker {worker_label}')
            response = await client.post(url + '/render', json={
                'source_url': source_url(video_path, ttl_seconds=int(timeout) + 3600),
                'source_codec': source_codec, 'duration': duration,
                'width': source_video['width'], 'height': source_video['height'],
                'pix_fmt': source_video['pix_fmt'], 'crf': crf, 'preset': preset,
                'filter_complex': filter_complex,
            })
            response.raise_for_status()
            acknowledgement = response.json()
            candidate = acknowledgement['id']
            if not isinstance(candidate, str) or not re.fullmatch('[a-f0-9]{32}', candidate):
                raise ValueError('Invalid worker render ID')
            identity = candidate
            if acknowledgement.get('filter_layout') != _REQUIRED_FILTER_LAYOUT:
                raise IncompatibleRenderWorker('worker is missing planar blur fix; rebuild worker image')
            while True:
                response = await client.get(f'{url}/render/{identity}')
                response.raise_for_status()
                status = response.json()
                state = status['state']
                fraction = float(status['progress'])
                if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                    raise ValueError('Invalid worker progress')
                maximum_progress = max(maximum_progress, fraction)
                if state in {'failed', 'cancelled'}:
                    raise ValueError('Worker could not complete whole-video render')
                if state not in {'queued', 'running', 'done'}:
                    raise ValueError('Unexpected worker render state')
                if on_progress:
                    await on_progress(maximum_progress * 0.98)
                if on_stage:
                    backend = status.get('backend', 'pending')
                    await on_stage(f'Remote whole-file blur — {backend}, {state} ({worker_label})')
                if state == 'done':
                    break
                await asyncio.sleep(_POLL_SECONDS)
            if on_stage:
                await on_stage('Downloading remote whole-file blur')
            async with client.stream('GET', f'{url}/render/{identity}/output') as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
            if not temporary.stat().st_size:
                raise ValueError('Worker returned an empty video')
            if on_stage:
                await on_stage('Validating remote video on server (no local blur encode)')
            # Includes the same decode checks as the local path. Validate
            # before caching: otherwise a failed remote output could poison
            # the whole-file retry's exists-means-complete checkpoint.
            await validate(temporary)
            temporary.replace(out_path)
            if on_progress:
                await on_progress(1.0)
            logger.info('Whole-file blur rendered remotely by %s', worker_label)
            return True
        try:
            return await asyncio.wait_for(execute(), timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never log request bodies or exception strings from worker data:
            # the source URL inside the JSON body is a bearer capability.
            if isinstance(exc, IncompatibleRenderWorker):
                reason = str(exc)  # Fixed local text, not untrusted response data.
            else:
                reason = f'HTTP {exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            logger.warning('Remote whole-file blur failed (%s); using server CPU', reason)
            if on_stage:
                await on_stage(f'Remote whole-file blur failed ({reason}) — server CPU fallback')
            return False
        finally:
            if identity:
                try:
                    await client.delete(f'{url}/render/{identity}', timeout=5)
                except Exception:
                    pass  # Lost workers must not prevent the local safety net.
            temporary.unlink(missing_ok=True)
