"""Short-lived, file-scoped capabilities for worker ffmpeg HTTP range reads.

No library path is accepted without a signature, and the token itself is never
sent to the worker. URLs grant read access to one exact file until expiry;
use a trusted network/TLS because the URL is itself a bearer credential.
"""
import hashlib
import hmac
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode


class SourceCapabilityLogFilter(logging.Filter):
    """Hide bearer query strings without mutating the ASGI request scope.

    Uvicorn's access record arguments are client, method, path, HTTP version,
    status. Redacting the log record, not scope['query_string'], is essential:
    the source endpoint still needs the original signature to authenticate.
    Reverse proxies must independently suppress these query strings too.
    """
    def filter(self, record):
        if isinstance(record.args, tuple) and len(record.args) == 5:
            path = record.args[2]
            if isinstance(path, str) and path.split('?', 1)[0] == '/internal/source':
                args = list(record.args)
                args[2] = '/internal/source?[redacted]'
                record.args = tuple(args)
        return True


def install_source_log_redaction():
    access = logging.getLogger('uvicorn.access')
    if not any(isinstance(f, SourceCapabilityLogFilter) for f in access.filters):
        access.addFilter(SourceCapabilityLogFilter())


def remote_source_enabled() -> bool:
    """Source transport is also usable by encoders without remote scanning."""
    return (bool(os.getenv('SPF_INTERNAL_BASE_URL', '').strip())
            and bool(os.getenv('SPF_INTERNAL_TOKEN', '').strip()))


def remote_extract_enabled() -> bool:
    """Require all transport settings; a partial configuration stays local."""
    return (os.getenv('SPF_REMOTE_EXTRACT') == '1'
            and bool(os.getenv('SPF_CLASSIFIER_URLS', '').strip())
            and remote_source_enabled())


def source_signature(path: str, expires: int) -> str:
    key = os.environ.get('SPF_INTERNAL_TOKEN', '').encode()
    return hmac.new(key, f'{expires}\n{path}'.encode(), hashlib.sha256).hexdigest()


def source_url(path: Path, ttl_seconds: int = 3600) -> str:
    """Render jobs need range-read capabilities longer than a scan chunk.
    Scope is still one exact file; callers choose a TTL bounded by their job
    deadline so a long GPU/CPU encode can issue later HTTP range requests.
    """
    if ttl_seconds <= 0:
        raise ValueError('Source capability lifetime must be positive')
    expires = int(time.time()) + ttl_seconds
    absolute = str(path.resolve())
    query = urlencode({'path': absolute, 'expires': expires,
                       'signature': source_signature(absolute, expires)})
    return os.environ['SPF_INTERNAL_BASE_URL'].rstrip('/') + '/internal/source?' + query
