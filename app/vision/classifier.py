"""Per-frame explicit-content classification via NudeNet, for scene detection.

NudeNet (not OpenNSFW2 -- see the scene-detection plan for why) is loaded lazily
and only inside functions, never at module import, mirroring
app.audio.forced_align's discipline exactly: every processing job that never
scans for scenes pays zero import/model-load cost for this being available.
"""

import asyncio
import itertools
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx

from app.vision.scene_cluster import FrameScore
from app.vision.remote_source import remote_extract_enabled, source_url

logger = logging.getLogger(__name__)

# --- Remote GPU classifier (DockerStacks patch) ---------------------------------
# SPF_CLASSIFIER_URLS: comma-separated http://host:port list of remote NudeNet
# workers (see downloadstation/vulgarr-worker). When set, every frame is sent
# over HTTP for GPU inference instead of running locally on CPU; on any remote
# failure (worker down, timeout, bad response) we transparently fall back to
# the original local CPU path so a powered-off desktop never breaks a scan.
# Frames are distributed round-robin across the listed URLs (e.g. one worker
# per GPU on the same desktop).

def _classifier_urls() -> list[str]:
    raw = os.environ.get("SPF_CLASSIFIER_URLS", "")
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


_remote_rr = itertools.count()
_remote_client: httpx.AsyncClient | None = None
_remote_client_lock = threading.Lock()


def _remote_http_client() -> httpx.AsyncClient:
    global _remote_client
    with _remote_client_lock:
        if _remote_client is None:
            _remote_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return _remote_client


async def _detect(frame_path: Path, local_detector_callable) -> list[dict]:
    """Classify one frame: remote GPU worker if configured, else (or on any
    remote failure) the local CPU NudeNet session. Always returns NudeNet's
    raw per-frame detection list (possibly empty)."""
    urls = _classifier_urls()
    if urls:
        url = urls[next(_remote_rr) % len(urls)]
        try:
            with frame_path.open("rb") as fh:
                resp = await _remote_http_client().post(
                    f"{url}/detect", files={"frame": (frame_path.name, fh, "image/jpeg")}
                )
            if resp.status_code == 200:
                return resp.json().get("detections", [])
            logger.warning("Remote classifier %s returned HTTP %d -- falling back to local", url, resp.status_code)
        except Exception:
            logger.warning("Remote classifier %s unreachable -- falling back to local CPU", url, exc_info=True)
    return await asyncio.to_thread(local_detector_callable, str(frame_path))
# --------------------------------------------------------------------------------

# NudeNet's 18-class taxonomy includes body parts that are just anatomically
# present in an ordinary frame (faces, feet, armpits, belly) alongside the
# classes that actually indicate nudity/exposure -- only the latter should drive
# a scene-detection score. Confirmed against a real spike run before committing
# to this set (see the scene-detection plan's Phase 0 findings).
_EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

_bundle: dict = {}
# classify_one (below) calls this from several concurrent asyncio.to_thread workers
# (scene_frame_classify_concurrency) -- an unguarded check-then-set here would let
# multiple threads race into constructing NudeDetector (and importing nudenet)
# simultaneously the first time a scan runs after container startup. Reproduced
# live for a sibling lazy-loader built the same unguarded way (a different model's
# first-import machinery got corrupted under exactly this race, manifesting as
# both a spurious ImportError and a bare SIGILL crash with no Python traceback) --
# applying the same lock-guarded fix here defensively rather than waiting to
# reproduce it against this specific model too.
_load_lock = threading.Lock()


def _get_detector():
    if "detector" not in _bundle:
        with _load_lock:
            if "detector" not in _bundle:  # re-check: another thread may have won the race while we waited
                from nudenet import NudeDetector

                _bundle["detector"] = NudeDetector()
    return _bundle["detector"]


def extract_frame(
    ffmpeg_bin: str, video_path: Path, timestamp: float, out_path: Path, max_long_edge: int | None = None
) -> None:
    """Blocking -- callers on the event loop must offload via asyncio.to_thread.
    Shared by the scan pipeline (below) and the scene-review thumbnail endpoint
    (app/routers/scenes.py). max_long_edge downscales (never upscales) before
    encoding -- used by app.vision.claude_verify to shrink the image-token cost
    of a Claude Vision call (Claude tokenizes images as ceil(w/28)*ceil(h/28)
    patches, so a 1920x1080 frame costs ~1560 tokens even after Anthropic's own
    auto-downscale; sending it pre-shrunk to ~512px costs closer to ~200).
    Left unset (None) for the thumbnail/review-UI callers, which want full
    quality for a human to actually look at."""
    cmd = [ffmpeg_bin, "-nostdin", "-loglevel", "error", "-ss", f"{timestamp:.3f}", "-i", str(video_path)]
    if max_long_edge is not None:
        cmd += ["-vf", f"scale='min({max_long_edge},iw)':'min({max_long_edge},ih)':force_original_aspect_ratio=decrease"]
    cmd += ["-frames:v", "1", "-q:v", "4", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg frame extraction failed at {timestamp:.2f}s in {video_path}")


def frame_confidence(detections: list[dict]) -> float:
    """Pure mapping from NudeNet's per-class detection list (each a dict with a
    "class" and "score") to a single confidence score -- the max score among
    classes that actually indicate exposure, 0.0 if none matched or the frame had
    no detections at all. Kept as a standalone function so it's testable without
    a real model."""
    scores = [d["score"] for d in detections if d.get("class") in _EXPLICIT_CLASSES]
    return max(scores, default=0.0)


# How much of a scan's total wall time the extraction phase accounts for, used
# only to blend a smooth combined progress bar across the two phases (not for
# scheduling). Measured directly on real homelab hardware against real
# NFS-mounted media: extracting 40 frames via one continuous decode took ~5.2s,
# classifying those same 40 already-extracted frames at concurrency=8 took
# ~3.0s -- extraction is the larger share of the two.
_EXTRACT_PROGRESS_WEIGHT = 0.6


async def _extract_frames_continuous(
    ffmpeg_bin: str,
    video_path: Path,
    frame_interval_seconds: float,
    out_dir: Path,
    duration_seconds: float,
    on_extract_progress=None,
    start_offset: float = 0.0,
    clip_duration: float | None = None,
) -> None:
    """One continuous sequential decode pass emitting a frame every
    frame_interval_seconds, replacing what used to be one independent
    `ffmpeg -ss <t>` re-seek per sampled timestamp.

    Confirmed via direct A/B benchmark against real homelab hardware and the
    same real (NFS-mounted) media file, at a cold/never-touched offset so
    neither side benefited from page-cache warmth: 40 individually re-seeked
    frames -- even run at concurrency=8 -- took ~26.6s; one continuous decode
    over the same 40s span took ~5.2s. Repeatedly re-seeking a multi-GB
    container is genuinely expensive here; one sequential read is not.
    `format=yuvj420p` works around an mjpeg-encoder failure ffmpeg throws on
    some full-range-YUV source segments otherwise ("Non full-range YUV is
    non-standard... ff_frame_thread_encoder_init failed"), found while
    benchmarking against a real mid-episode offset.

    start_offset/clip_duration scope the decode to a short window instead of
    the whole file -- used by scan_window_frames for the dense per-candidate
    verification pass, where re-decoding the entire movie just to resample a
    single 20-40s scene would defeat the point of it being cheap. A single
    `-ss` before `-i` still costs one seek either way, but that's one seek per
    candidate (a handful per scan), not one per sampled frame.
    """
    cmd = [ffmpeg_bin, "-nostdin", "-loglevel", "error", "-progress", "pipe:1"]
    if start_offset > 0:
        cmd += ["-ss", f"{start_offset:.3f}"]
    cmd += ["-i", str(video_path)]
    if clip_duration is not None:
        cmd += ["-t", f"{clip_duration:.3f}"]
    cmd += [
        "-vf", f"fps=1/{frame_interval_seconds},format=yuvj420p",
        "-q:v", "4",
        str(out_dir / "frame_%08d.jpg"),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stderr_chunks: list[bytes] = []

    async def read_stderr() -> None:
        assert proc.stderr is not None
        async for chunk in proc.stderr:
            stderr_chunks.append(chunk)

    async def read_progress() -> None:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    current = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                if on_extract_progress is not None and duration_seconds > 0:
                    await on_extract_progress(max(0.0, min(1.0, current / duration_seconds)))

    try:
        await asyncio.gather(read_stderr(), read_progress(), proc.wait())
    except asyncio.CancelledError:
        # Same reasoning as remux.py's _run_with_progress -- cancellation must
        # kill the actual ffmpeg process, not just stop awaiting it.
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg continuous frame extraction failed for {video_path}: "
            f"{b''.join(stderr_chunks).decode(errors='replace')[:500]}"
        )


def _load_resume_scores(path: Path) -> list[FrameScore]:
    """One JSON object per line ({"t": timestamp, "c": confidence}), tolerating
    a corrupt/truncated trailing line -- a process killed mid-write (e.g. this
    app's own container being rebuilt mid-scan) can leave one partial line at
    the end of an append-only file; skip it rather than fail the whole load."""
    if not path.exists():
        return []
    scores: list[FrameScore] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            scores.append(FrameScore(timestamp=float(obj["t"]), confidence=float(obj["c"])))
        except Exception:
            continue
    return scores


def _resumable_prefix(
    scores: list[FrameScore], start_offset: float, frame_interval_seconds: float
) -> tuple[float, list[FrameScore]]:
    """How many seconds of *contiguous* coverage, starting at start_offset,
    the given (possibly out-of-order, possibly gappy) scores actually
    represent, and the subset that falls within that trusted run -- pure
    function, testable without any real scan.

    Frame classification runs at concurrency > 1 (see _scan_frames), so
    completions -- and therefore what gets checkpointed to a resume file --
    can land out of order. If a prior attempt was interrupted right after a
    later-timestamped frame happened to finish first, trusting a raw
    max(timestamp) would silently skip re-checking earlier frames that were
    still in flight and never got written at all -- a real, silent detection
    gap, not just wasted work (exactly the kind of bug this whole feature has
    spent a lot of effort avoiding elsewhere: a missed frame near a scene
    boundary is indistinguishable from the classifier just not seeing
    anything there). This walks the sorted timestamps from start_offset and
    stops at the first real gap (more than one sample interval, with a little
    float-rounding slack), so a resume only ever trusts a genuinely unbroken
    run from the very start and re-does everything from the first hole
    onward -- always safe (worst case, a little redundant reclassification
    right at the resume boundary), never silently lossy."""
    if not scores:
        return 0.0, []
    ordered = sorted((s for s in scores if s.timestamp >= start_offset - 1e-6), key=lambda s: s.timestamp)
    if not ordered:
        return 0.0, []
    tolerance = frame_interval_seconds * 1.5
    trusted: list[FrameScore] = []
    expected = start_offset
    for s in ordered:
        if s.timestamp > expected + tolerance:
            break
        trusted.append(s)
        expected = max(expected, s.timestamp)
    if not trusted:
        return 0.0, []
    return (expected - start_offset) + frame_interval_seconds, trusted


# Bounds how much extraction work an interruption can ever lose on the
# resumable whole-file path (see _scan_frames): extraction runs as one
# continuous ffmpeg decode pass with no progress checkpointed until it
# finishes and classification of its frames begins, so a scan done as a
# single pass over a multi-hour file has no real resumability at all -- a
# restart mid-extraction (confirmed live on a real 2h11m 4K/HEVC title: still
# zero checkpointed frames after several minutes, all of it inside one
# extraction pass covering the *entire remaining* duration) would still lose
# everything back to the last successfully *classified* frame, which could be
# the whole file. Chunking bounds the loss to at most one chunk's worth
# regardless of total file length. 5 minutes is a lot more decode work than
# one seek's cost is worth worrying about -- this codebase's own existing
# per-candidate windowed scan (scan_window_frames) already accepts one `-ss`
# seek as cheap relative to a handful of seconds of decode, and 5 minutes is
# far more decode than that per seek paid for here.
_SCAN_CHUNK_SECONDS = 300.0


async def _scan_frames(
    video_path: Path,
    ffmpeg_bin: str,
    frame_interval_seconds: float,
    scan_duration_seconds: float,
    on_progress=None,
    concurrency: int = 1,
    start_offset: float = 0.0,
    clip_duration: float | None = None,
    resume_scores_path: Path | None = None,
    on_status=None,
) -> list[FrameScore]:
    """Shared implementation behind scan_video_frames (whole file) and
    scan_window_frames (one short padded clip) -- extract via one continuous
    decode pass per chunk, classify each sampled frame, and return one
    FrameScore per successfully classified frame, timestamped relative to the
    real file (not the clip) via start_offset. A frame that fails to classify
    is skipped (logged, not fatal) -- same fall-back-gracefully philosophy as
    align_matches_for_cue's per-cue try/except. If extraction of a chunk fails
    outright, returns whatever was already classified (and, for the resumable
    path, already checkpointed) rather than raising -- callers already treat
    a short/empty result as "found less," and a truly broken/corrupt source
    shouldn't crash the whole job.

    concurrency > 1 runs multiple already-extracted frames' classification in
    flight at once via a bounded semaphore -- each is an independent
    onnxruntime call with no shared mutable state (the detector's a single
    loaded session, safe for concurrent .run() calls), so this is pure
    parallelism with no effect on which candidates come out the other end, only
    how long it takes to get there.

    resume_scores_path, when given (only scan_video_frames' whole-file caller
    does -- scan_window_frames' per-candidate passes are already cheap enough
    not to need this), makes the scan resumable across a process restart: the
    remaining span is processed in _SCAN_CHUNK_SECONDS chunks (extract, then
    classify+checkpoint that chunk, before starting the next), each
    classified frame appended to the file as an independent JSON line as soon
    as it completes, and a prior run's *trusted contiguous* checkpoint (see
    _resumable_prefix -- concurrent completion means the raw file can have
    out-of-order or gappy entries, which this does not naively trust) is
    loaded and skipped past on the next call. Without chunking, a scan
    interrupted mid-extraction would have nothing checkpointed at all yet for
    however long that one pass over the *entire* remaining duration takes --
    confirmed live on a real multi-hour file, see _SCAN_CHUNK_SECONDS."""
    expected_total = max(1, round(scan_duration_seconds / frame_interval_seconds))

    resumed_scores: list[FrameScore] = []
    resume_seconds = 0.0
    if resume_scores_path is not None:
        raw_scores = _load_resume_scores(resume_scores_path)
        resume_seconds, resumed_scores = _resumable_prefix(raw_scores, start_offset, frame_interval_seconds)
        if resumed_scores:
            logger.info(
                "Resuming scene scan for %s: %d frame(s) already checkpointed (%.1fs covered)",
                video_path, len(resumed_scores), resume_seconds,
            )
        # Rewrite the checkpoint file to hold exactly the trusted prefix --
        # drops any untrusted/gappy tail from an interrupted attempt (see
        # _resumable_prefix) so newly (re)classified frames append cleanly
        # after it, instead of piling up alongside stale/ambiguous entries.
        if raw_scores:
            resume_scores_path.write_text(
                "".join(json.dumps({"t": s.timestamp, "c": s.confidence}) + "\n" for s in resumed_scores)
            )

    effective_start_offset = start_offset + resume_seconds
    full_span = clip_duration if clip_duration is not None else scan_duration_seconds
    remaining_duration = full_span - resume_seconds
    if remaining_duration <= 1e-6:
        return resumed_scores

    already_done_units = round(resume_seconds / frame_interval_seconds)
    remaining_units = expected_total - already_done_units
    extract_units = round(remaining_units * _EXTRACT_PROGRESS_WEIGHT)
    classify_units = max(1, remaining_units - extract_units)
    total_frames_remaining = max(1, round(remaining_duration / frame_interval_seconds))

    # Only the resumable whole-file path chunks -- scan_window_frames (no
    # resume_scores_path) keeps its original single-pass behavior exactly,
    # since its clips are already short enough not to need this.
    chunk_size = min(_SCAN_CHUNK_SECONDS, remaining_duration) if (resume_scores_path is not None or remote_extract_enabled()) else remaining_duration

    # Local detector is loaded lazily: when SPF_CLASSIFIER_URLS is set and the
    # workers stay healthy, nudenet/onnxruntime is never imported on this host
    # at all (patch: keeps the server CPU free during remote scans).
    detector_holder: dict = {}

    def _local_detect(path: str):
        det = detector_holder.get("det")
        if det is None:
            det = _get_detector()
            detector_holder["det"] = det
        return det.detect(path)
    scores: list[FrameScore] = list(resumed_scores)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    # Callers' on_progress (see app.queue.scene_worker.on_scan_progress) commits
    # against a single AsyncSession shared across the whole job -- SQLAlchemy's
    # async sessions reject concurrent operations outright ("This session is
    # provisioning a new connection..."). classify_one runs concurrently now
    # (that's the whole point), so multiple completions can land in the same
    # event-loop tick; this lock serializes the on_progress call and the
    # resume-checkpoint file append together, not the actual extract+
    # classify work, so it doesn't undo the parallelism.
    progress_lock = asyncio.Lock()
    completed_overall = 0
    # Seconds of remaining_duration whose extraction has *fully* completed,
    # across all prior chunks -- updated only after a chunk's extraction
    # finishes, so classify_one's progress contribution (below) reflects how
    # much extraction is actually done so far, not the whole scan's total
    # extraction budget the moment any classification starts at all (a real
    # bug caught live: reusing the single-pass formula's `+ extract_units`
    # unconditionally made progress jump straight to ~60% as soon as the
    # first chunk's classification began, since extract_units there meant
    # "all extraction, ever," not "this chunk's share").
    extraction_seconds_done = 0.0

    # A bounded producer owns tempdirs until the consumer has checkpointed
    # their frames. Two slots mean exactly current + next, not an unbounded
    # decode backlog on a fast disk / slow inference box. Cancellation joins
    # the producer before deleting its files (including ffmpeg cancellation).
    prepared = asyncio.Queue()
    slots = asyncio.Semaphore(2)
    owned_dirs = []
    chunk_count = max(1, math.ceil(remaining_duration / chunk_size))
    extracted_fraction = 0.0
    remote_ok = remote_fallback = in_flight = 0
    started = time.monotonic()
    progress_high_water = already_done_units

    async def report(phase):
        nonlocal progress_high_water
        async with progress_lock:
            done = already_done_units + round(extracted_fraction * extract_units) + round(
                completed_overall / total_frames_remaining * classify_units
            )
            progress_high_water = max(progress_high_water, min(done, expected_total))
            if on_status is not None:
                fps = completed_overall / max(0.001, time.monotonic() - started)
                await on_status(f"{phase} · remote {remote_ok} · fallback {remote_fallback} · {fps:.1f} fps · {in_flight} in flight")
            if on_progress is not None:
                await on_progress(progress_high_water, expected_total)

    async def produce():
        nonlocal extracted_fraction, remote_ok, remote_fallback
        offset = 0.0
        try:
            for number in range(chunk_count):
                await slots.acquire()
                duration = min(chunk_size, remaining_duration - offset)
                remote_scores = None
                if remote_extract_enabled():
                    await report(f"Extracting chunk {number + 1}/{chunk_count} (remote)")
                    try:
                        urls = _classifier_urls()
                        url = urls[next(_remote_rr) % len(urls)]
                        response = await _remote_http_client().post(
                            url + '/scan', timeout=660,
                            json={'source_url': source_url(video_path),
                                  'start': effective_start_offset + offset,
                                  'duration': duration, 'interval': frame_interval_seconds},
                        )
                        response.raise_for_status()
                        remote_scores = response.json()['scores']
                        count = max(1, round(duration / frame_interval_seconds))
                        if len(remote_scores) != count:
                            raise ValueError('Incomplete remote scan')
                        for i, score in enumerate(remote_scores):
                            expected = effective_start_offset + offset + i * frame_interval_seconds
                            if not math.isfinite(float(score['t'])) or abs(float(score['t']) - expected) > 1e-5 or not 0 <= float(score['c']) <= 1:
                                raise ValueError('Invalid remote score')
                        remote_ok += count
                        extracted_fraction = (offset + duration) / remaining_duration
                        await prepared.put((None, remote_scores))
                        offset += duration
                        continue
                    except Exception:
                        # Do not log exception text: httpx errors contain the
                        # request URL, and source capabilities are credentials.
                        logger.warning('Remote extraction failed; retrying local extraction')
                        remote_fallback += max(1, round(duration / frame_interval_seconds))
                tmp = tempfile.TemporaryDirectory()
                owned_dirs.append(tmp)
                async def extraction_progress(fraction):
                    nonlocal extracted_fraction
                    extracted_fraction = (offset + fraction * duration) / remaining_duration
                    await report(f"Extracting chunk {number + 1}/{chunk_count}")
                async def extract():
                    await _extract_frames_continuous(
                        ffmpeg_bin, video_path, frame_interval_seconds, Path(tmp.name), duration,
                        on_extract_progress=extraction_progress,
                        start_offset=effective_start_offset + offset, clip_duration=duration,
                    )
                if remote_extract_enabled():
                    # Remote scans bypass the outer heavy lock. A fallback
                    # must rejoin audio/blur exclusion before touching NFS.
                    from app.queue.heavy_pipeline_lock import HEAVY_PIPELINE_LOCK
                    async with HEAVY_PIPELINE_LOCK:
                        await extract()
                else:
                    await extract()
                extracted_fraction = (offset + duration) / remaining_duration
                await prepared.put((tmp, effective_start_offset + offset))
                offset += duration
        except Exception as exc:
            await prepared.put(exc)
        finally:
            await prepared.put(None)

    async def classify_batch(items, chunk_abs_start):
        nonlocal completed_overall, remote_ok, remote_fallback, in_flight
        async with semaphore:
            in_flight += 1
            try:
                results = None
                urls = _classifier_urls()
                if urls:
                    try:
                        url = urls[next(_remote_rr) % len(urls)]
                        response = await _remote_http_client().post(
                            f"{url}/detect-batch",
                            files=[('frames', (p.name, p.read_bytes(), 'image/jpeg')) for _, p in items],
                        )
                        response.raise_for_status()
                        results = response.json()['detections']
                        if not isinstance(results, list) or len(results) != len(items) or any(not isinstance(r, list) for r in results):
                            raise ValueError('incomplete batch response')
                        # Validate before trusting any part of the response.
                        for result in results:
                            frame_confidence(result)
                        remote_ok += len(items)
                    except Exception:
                        logger.warning('Batch classifier failed; retrying individual frames', exc_info=True)
                        results = None
                for position, (index, path) in enumerate(items):
                    ts = chunk_abs_start + index * frame_interval_seconds
                    try:
                        if results is None:
                            detections = await _detect(path, _local_detect)
                            if urls:
                                remote_fallback += 1
                        else:
                            detections = results[position]
                        score = FrameScore(timestamp=ts, confidence=frame_confidence(detections))
                        async with progress_lock:
                            scores.append(score)
                            if resume_scores_path is not None:
                                with resume_scores_path.open('a') as fh:
                                    fh.write(json.dumps({'t': ts, 'c': score.confidence}) + '\n')
                    except Exception:
                        logger.warning('Scene frame failed at %.2fs', ts, exc_info=True)
                    completed_overall += 1
                    await report(f"Classifying {already_done_units + completed_overall}/{expected_total} frames")
            finally:
                in_flight -= 1

    producer = asyncio.create_task(produce())
    try:
        while True:
            item = await prepared.get()
            if item is None:
                break
            if isinstance(item, Exception):
                logger.warning('Chunk extraction failed: %s', item)
                break
            tmp, chunk_abs_start = item
            if tmp is None:
                for raw in chunk_abs_start:
                    score = FrameScore(timestamp=float(raw['t']), confidence=float(raw['c']))
                    async with progress_lock:
                        scores.append(score)
                        if resume_scores_path is not None:
                            with resume_scores_path.open('a') as fh:
                                fh.write(json.dumps({'t': score.timestamp, 'c': score.confidence}) + '\n')
                    completed_overall += 1
                    await report(f"Classifying {already_done_units + completed_overall}/{expected_total} frames")
                slots.release()
                continue
            frames = list(enumerate(sorted(Path(tmp.name).glob('frame_*.jpg'))))
            batch_size = 12 if _classifier_urls() else 1
            await asyncio.gather(*(classify_batch(frames[i:i + batch_size], chunk_abs_start)
                                   for i in range(0, len(frames), batch_size)))
            tmp.cleanup()
            owned_dirs.remove(tmp)
            slots.release()
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        for tmp in owned_dirs:
            tmp.cleanup()
    return scores


async def scan_video_frames(
    video_path: Path, ffmpeg_bin: str, duration_seconds: float,
    frame_interval_seconds: float, on_progress=None, concurrency: int = 1,
    resume_scores_path: Path | None = None, on_status=None,
) -> list[FrameScore]:
    """Sample the whole title, preserving the contiguous resume prefix."""
    return await _scan_frames(
        video_path, ffmpeg_bin, frame_interval_seconds, duration_seconds,
        on_progress=on_progress, concurrency=concurrency,
        resume_scores_path=resume_scores_path, on_status=on_status,
    )


async def scan_window_frames(
    video_path: Path, ffmpeg_bin: str, window_start: float, window_end: float,
    pad_seconds: float, frame_interval_seconds: float, concurrency: int = 1,
) -> list[FrameScore]:
    """Densely sample a padded candidate window on the source timeline."""
    clip_start = max(0.0, window_start - pad_seconds)
    clip_duration = (window_end - window_start) + 2 * pad_seconds
    return await _scan_frames(
        video_path, ffmpeg_bin, frame_interval_seconds, clip_duration,
        concurrency=concurrency, start_offset=clip_start, clip_duration=clip_duration,
    )

