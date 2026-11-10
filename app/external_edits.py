"""Compose audio and scene edits for separate libraries without touching sources.

Retain only an audio-only master, not another full video copy. Every scene
Apply then reads the original video and this master, so re-approving/rejecting
scenes can remove earlier blur/mute windows without removing profanity edits.
An audio reprocess builds a fresh master from the original, then regenerates
any approved scenes before replacing the visible edit. The caller holds the
same audio/blur heavy gate throughout this operation.
"""
import asyncio
import datetime
import logging
import math
import uuid
from pathlib import Path

from sqlalchemy import select

from app.config import settings as app_settings
from app.db.models import DetectedScene, SceneReviewStatus
from app.domain import parse_index_list, serialize_index_list
from app.mux.remux import RemuxError, _audio_streams, _run, probe
from app.mux.tempfiles import remove_orphaned_temp_files
from app.output import publish_file, source_fingerprint
from app.processing import process_video
from app.scenes.pipeline import apply_scene_blur

logger = logging.getLogger(__name__)


def _sweep_stale_work(source: Path, destination: Path, cache_dir: Path, retained: str | None):
    """Caller holds the heavy gate: no other audio/blur writer is active.
    Remove only our exact title's hidden staging names and unreferenced audio
    masters in its private .vulgarr directory. This bounds leftovers from process
    death, which never reaches a coroutine's finally cleanup.
    """
    remove_orphaned_temp_files(destination.with_name(source.name), min_age_seconds=0)
    for candidate in cache_dir.glob('*.mka'):
        if retained and candidate.resolve() == Path(retained).resolve():
            continue
        try:
            candidate.unlink()
        except OSError:
            logger.warning('Could not remove orphaned audio master %s', candidate)


async def _cache_audio(video: Path, cache: Path) -> None:
    """Copy all original + clean tracks with their metadata; never re-encode
    real audio. Append a tiny silent timing anchor (excluded from all published
    maps) so an initial audio delay doesn't become a new container time origin.

    Keep the same relative audio-stream order as the edited video, because
    Title.clean_track_audio_indices describes that order, not absolute stream
    indices in this audio-only container. Validate before exposing the cache.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    staged = cache.with_name(cache.stem + '.tmp.mka')
    try:
        source = await probe(app_settings.ffprobe_bin, video)
        original_audio = _audio_streams(source)
        if not original_audio:
            raise RemuxError('No audio streams to retain')
        anchor_index = len(original_audio)
        code, _, err = await _run([
            app_settings.ffmpeg_bin, '-y', '-nostdin', '-loglevel', 'error',
            '-i', str(video), '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono:d=0.02',
            '-map', '0:a', '-map', '1:a:0', '-c:a', 'copy',
            f'-c:a:{anchor_index}', 'pcm_s16le',
            f'-metadata:s:a:{anchor_index}', 'VULGARR_TIMING_ANCHOR=1',
            f'-disposition:a:{anchor_index}', '0', '-map_metadata', '0',
            '-avoid_negative_ts', 'disabled', '-vn', '-sn', str(staged),
        ], timeout=14400)
        if code:
            raise RemuxError(f'Could not retain clean audio master: {err[-2000:]}')
        cached = await probe(app_settings.ffprobe_bin, staged)
        copied_audio = _audio_streams(cached)
        if (len(copied_audio) != len(original_audio)
                or len(cached['streams']) != len(original_audio) + 1
                or cached['streams'][-1] in copied_audio
                or any(s['codec_type'] != 'audio' for s in cached['streams'])):
            raise RemuxError('Clean audio master stream count mismatch')
        # An audio-only file otherwise starts at its first real packet: if
        # dialogue starts at +5s, ffmpeg would subtract those 5s on the next
        # read and break sync. The last, tagged 20ms silent track anchors t=0
        # without re-encoding any real audio. _audio_streams excludes it and
        # all published maps take only the first N real tracks.
        start = float(cached['format'].get('start_time', 0))
        if not math.isfinite(start) or abs(start) > 0.01:
            raise RemuxError('Clean audio master timeline shifted')
        for original, copied in zip(original_audio, copied_audio):
            if original.get('start_time') is not None and copied.get('start_time') is not None:
                expected = float(original['start_time']) - float(source['format'].get('start_time', 0))
                actual = float(copied['start_time'])
                if not math.isfinite(expected) or not math.isfinite(actual) or abs(actual - expected) > 0.01:
                    raise RemuxError('Clean audio master stream timing shifted')
            for key in ('codec_name', 'sample_rate', 'channels'):
                if original.get(key) != copied.get(key):
                    raise RemuxError(f'Clean audio master {key} mismatch')
            # Audio can legitimately start late/end before the video (silent
            # credits, for example). Comparing an audio-only container's
            # duration to the whole movie would incorrectly reject that file.
            # Compare individual stream durations only when both report them;
            # Matroska often omits them, so ffmpeg success + stream identity
            # and count are the fallback integrity checks in that case.
            if original.get('duration') is not None and copied.get('duration') is not None:
                durations = (float(copied['duration']), float(original['duration']))
                if not all(math.isfinite(d) for d in durations) or abs(durations[0] - durations[1]) > 2:
                    raise RemuxError('Clean audio master duration mismatch')
        staged.replace(cache)
    finally:
        staged.unlink(missing_ok=True)


async def process_external_video(session, *, title, job_id: int, destination: Path, **options):
    """Publish only once both edits succeed; keep the previous edit on failure.

    Settings changes affect future jobs, not this job's pinned destination.
    Unique, per-attempt cache names also leave a previous successful master
    intact if a worker is interrupted while writing its replacement.
    """
    source = Path(title.video_path)
    fingerprint = source_fingerprint(source)
    identity = uuid.uuid4().hex
    staged = destination.with_name(f'.{source.stem}.spf-tmp.audio-{job_id}-{identity}{source.suffix}')
    # Keep persistent, potentially multi-GB audio on the selected library
    # filesystem, not the usually much smaller application-data volume. This
    # hidden directory contains audio-only .mka files, never library videos.
    cache = destination.parent / '.vulgarr' / 'clean_audio' / str(title.id) / f'{identity}.mka'
    previous_cache = title.clean_audio_path
    previous_fingerprint = title.clean_audio_source_fingerprint
    previous_source_indices = getattr(title, 'source_clean_track_audio_indices', None)
    if previous_cache:
        source_indices = (parse_index_list(previous_source_indices)
                          if fingerprint == previous_fingerprint else None)
    else:
        source_indices = options.get('known_clean_indices')
    installed = False
    approved_query = select(DetectedScene.id).where(
        DetectedScene.title_id == title.id,
        DetectedScene.status == SceneReviewStatus.approved,
    ).limit(1)
    approved = await session.scalar(approved_query)
    progress = options.get('on_progress')
    stage = options.get('on_stage')
    audio_share = 0.45 if approved is not None else 0.85

    async def audio_progress(fraction):
        if progress is not None:
            await progress(audio_share * fraction)

    async def blur_progress(fraction):
        if progress is not None:
            base = audio_share + 0.05
            await progress(base + (0.95 - base) * fraction)

    try:
        await asyncio.to_thread(_sweep_stale_work, source, destination, cache.parent, previous_cache)
        # These indices describe the *edited* copy. The untouched source can
        # have fewer tracks; passing destination indices into its audio plan
        # would drop unrelated source tracks or fail stream mapping.
        options['known_clean_indices'] = source_indices
        options['on_progress'] = audio_progress
        outcome = await process_video(session, output_path=staged, **options)
        if stage is not None:
            await stage('Retaining clean audio for future scene edits')
        await _cache_audio(staged, cache)
        if source_fingerprint(source) != fingerprint:
            raise RemuxError('Source video changed during processing; retry the job')
        # Reviews can change during a long audio pass. Use the current set
        # when deciding whether a combined rebuild is still needed.
        approved = await session.scalar(approved_query)
        if approved is not None:
            # Only the audio master is needed from this full-video staging
            # file. Release its space before building the blurred replacement,
            # rather than keeping two new multi-GB video outputs alive at once.
            staged.unlink()
            # Audio and scene jobs use different ID spaces. Prefix the work
            # directory key to avoid inheriting an unrelated SceneJob's work.
            await apply_scene_blur(session, title, f'audio-{job_id}-{identity}',
                output_path=destination, on_progress=blur_progress,
                on_stage=stage, commit=False, audio_source_path=cache)
        else:
            if stage is not None:
                await stage('Publishing Vulgarr Edit')
            await publish_file(staged, destination)
            title.vulgarr_edit_path = str(destination)
            title.vulgarr_edit_generated_at = datetime.datetime.utcnow()
            stale = await session.execute(select(DetectedScene).where(
                DetectedScene.title_id == title.id, DetectedScene.applied_at.is_not(None)))
            for scene in stale.scalars():
                scene.applied_at = None
        # Delay Title mutation until after blur and its progress callbacks:
        # those callbacks commit this shared AsyncSession, not just job fields.
        # A failed encode must never expose a pending master as completed.
        title.clean_audio_path = str(cache)
        title.clean_audio_source_fingerprint = fingerprint
        title.source_clean_track_audio_indices = serialize_index_list(source_indices) if source_indices else None
        # A committed cache reference must precede deletion of the previous
        # master; a crash must never leave the DB pointing at a deleted file.
        await session.commit()
        installed = True
        if progress is not None:
            await progress(1.0)
        if previous_cache and previous_cache != str(cache):
            try:
                Path(previous_cache).unlink(missing_ok=True)
            except OSError:
                pass  # The successful edit must not fail over cache cleanup.
        return outcome
    finally:
        staged.unlink(missing_ok=True)
        if not installed:
            title.clean_audio_path = previous_cache
            title.clean_audio_source_fingerprint = previous_fingerprint
            title.source_clean_track_audio_indices = previous_source_indices
            cache.unlink(missing_ok=True)
