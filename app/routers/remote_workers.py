"""Worker-only source transport and human-facing, independently polled health.

The pinned Starlette predates FileResponse range support, so serve a single
RFC 7233 byte range explicitly. Reads are bounded and thread-offloaded by
StreamingResponse, never buffered as a whole multi-GB title in application RAM.
"""
import asyncio
import hmac
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from app.vision.classifier import _classifier_urls
from app.vision.remote_source import source_signature

router = APIRouter()
templates = Jinja2Templates(directory='app/templates')


def byte_range(header: str | None, size: int) -> tuple[int, int]:
    """One inclusive range; invalid/unsatisfiable ranges must never read a file."""
    if not header:
        return 0, size - 1
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError('Invalid range')
    first, last = match.groups()
    if first:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    else:
        length = int(last)
        if length <= 0:
            raise ValueError('Invalid suffix')
        start, end = max(0, size - length), size - 1
    if start > end or start >= size:
        raise ValueError('Unsatisfiable range')
    return start, end


@router.api_route('/internal/source', methods=['GET', 'HEAD'])
async def source(request: Request, path: str, expires: int, signature: str):
    if (not os.getenv('SPF_INTERNAL_TOKEN') or expires < time.time()
            or not hmac.compare_digest(signature, source_signature(path, expires))):
        raise HTTPException(403, 'Invalid or expired source capability')
    file = Path(path)
    try:
        size = file.stat().st_size
        if not file.is_file():
            raise OSError()
    except OSError:
        raise HTTPException(404, 'Source unavailable')
    try:
        start, end = byte_range(request.headers.get('range'), size)
    except ValueError:
        raise HTTPException(416, headers={'Content-Range': f'bytes */{size}'})
    headers = {'Accept-Ranges': 'bytes', 'Content-Length': str(max(0, end - start + 1)),
               'Cache-Control': 'no-store'}
    status = 206 if 'range' in request.headers else 200
    if status == 206:
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'

    def chunks():
        if request.method == 'HEAD':
            return
        with file.open('rb') as handle:
            handle.seek(start)
            left = end - start + 1
            while left > 0:
                data = handle.read(min(left, 1024 * 1024))
                if not data:
                    break
                left -= len(data)
                yield data
    return StreamingResponse(chunks(), status_code=status, headers=headers,
                             media_type='application/octet-stream')


@router.get('/queue/workers')
async def workers(request: Request):
    async def ping(client, url):
        try:
            health = await client.get(url + '/health')
            health.raise_for_status()
            response = await client.get(url + '/stats')
            response.raise_for_status()
            return {'url': url, 'online': True, **response.json()}
        except Exception:
            return {'url': url, 'online': False}
    async with httpx.AsyncClient(timeout=2.0) as client:
        rows = await asyncio.gather(*(ping(client, url) for url in _classifier_urls()))
    return templates.TemplateResponse('partials/worker_status.html',
                                      {'request': request, 'workers': rows})
