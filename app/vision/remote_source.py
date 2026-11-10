"""Short-lived, file-scoped capabilities for worker ffmpeg HTTP range reads.

No library path is accepted without a signature, and the token itself is never
sent to the worker. URLs grant read access to one exact file until expiry;
use a trusted network/TLS because the URL is itself a bearer credential.
"""
import hashlib
import hmac
import os
import time
from pathlib import Path
from urllib.parse import urlencode


def remote_extract_enabled() -> bool:
    """Require all transport settings; a partial configuration stays local."""
    return (os.getenv('SPF_REMOTE_EXTRACT') == '1'
            and bool(os.getenv('SPF_CLASSIFIER_URLS', '').strip())
            and bool(os.getenv('SPF_INTERNAL_BASE_URL', '').strip())
            and bool(os.getenv('SPF_INTERNAL_TOKEN', '').strip()))


def source_signature(path: str, expires: int) -> str:
    key = os.environ.get('SPF_INTERNAL_TOKEN', '').encode()
    return hmac.new(key, f'{expires}\n{path}'.encode(), hashlib.sha256).hexdigest()


def source_url(path: Path) -> str:
    expires = int(time.time()) + 3600
    absolute = str(path.resolve())
    query = urlencode({'path': absolute, 'expires': expires,
                       'signature': source_signature(absolute, expires)})
    return os.environ['SPF_INTERNAL_BASE_URL'].rstrip('/') + '/internal/source?' + query
