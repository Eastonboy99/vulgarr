"""Destination policy and atomic publication shared by audio and scene edits.

Configured roots describe paths *inside* the vulgarr container. We mirror the
source relative to SPF_MEDIA_ROOT rather than flattening filenames: two shows
can both contain episode.mkv, and two movies can both contain movie.mkv. Files
on an additional mount outside that root use an explicit _external subtree,
retaining their absolute parent hierarchy instead of risking a collision.
"""
import asyncio
import errno
import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path

from app.db.session import get_setting

logger = logging.getLogger(__name__)


def edit_destination(source: Path, media_root: Path, output_root: str) -> Path | None:
    """Blank means legacy behavior; configured destinations never name source."""
    if not output_root.strip():
        return None
    root = Path(output_root.strip()).expanduser()
    if not root.is_absolute():
        raise ValueError('Edited video output folders must be absolute container paths')
    # Mirror the library's logical hierarchy, not a file symlink's target
    # hierarchy. Resolve symlinks only for the overwrite-safety check below.
    source = Path(os.path.abspath(source))
    try:
        relative = source.relative_to(Path(os.path.abspath(media_root)))
    except ValueError:
        anchor = source.anchor.replace(':', '').strip('/\\').replace('\\', '/')
        relative = Path('_external') / anchor / source.relative_to(source.anchor)
    result = root / relative.with_name(f'{relative.stem} - Vulgarr Edit{relative.suffix}')
    if result.resolve() == source.resolve():
        raise ValueError('Edited output must not overwrite the source video')
    return result


async def title_edit_destination(session, title, media_root: Path) -> Path | None:
    kind = getattr(title.media_type, 'value', title.media_type)
    key = 'movie_output_folder' if kind == 'movie' else 'tv_output_folder'
    root = await get_setting(session, key) or ''
    return edit_destination(Path(title.video_path), media_root, root)


def source_fingerprint(source: Path) -> str:
    stat = source.stat()
    return hashlib.sha256(repr((str(source.resolve()), stat.st_size, stat.st_mtime_ns)).encode()).hexdigest()


async def publish_file(staged: Path, destination: Path) -> None:
    """Publish atomically even when source staging and destination are on different
    mounts. A cross-device move is NOT atomic: copy into a hidden temp on the
    destination filesystem first, then rename there. A failed copy leaves the
    previous edit intact. Join shielded file I/O on cancellation before cleanup;
    cancelling to_thread alone cannot stop its underlying OS copy.
    """
    if staged.resolve() == destination.resolve():
        raise ValueError('Staging file and destination must differ')

    def publish():
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            staged.replace(destination)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        tmp = destination.with_name(f'.{destination.stem}.spf-tmp.{uuid.uuid4().hex}{destination.suffix}')
        try:
            with staged.open('rb') as source, tmp.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            tmp.replace(destination)
            try:
                staged.unlink()
            except OSError:
                logger.warning('Published edit but could not remove staging file %s', staged)
        finally:
            tmp.unlink(missing_ok=True)

    task = asyncio.create_task(asyncio.to_thread(publish))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # Publication may have completed, just as with the original audio
        # backup/swap. In either case there must never be a half-written edit.
        await task
        raise
