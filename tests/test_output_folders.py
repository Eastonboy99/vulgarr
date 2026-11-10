"""Separate-library destinations, source preservation, and composition regressions.

ffmpeg is mocked at the boundary: these tests verify path policy, stream mapping,
atomic publication and pipeline orchestration without requiring media or CUDA.
"""
import asyncio
import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import output
from app.domain import MediaType, Severity
from app.mux.remux import RemuxError


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize('kind,relative', [
    (MediaType.movie, 'Movies/Film (2024)/Film.mkv'),
    (MediaType.episode, 'TV/Show/Season 01/Show - S01E01.mkv'),
])
def test_mirror_source_hierarchy_and_distinct_roots(monkeypatch, tmp_path, kind, relative):
    media = tmp_path / 'media'
    source = media / relative
    title = SimpleNamespace(media_type=kind, video_path=str(source))
    roots = {'movie_output_folder': str(tmp_path/'movie-edits'),
             'tv_output_folder': str(tmp_path/'tv-edits')}
    async def setting(session, key):
        return roots[key]
    monkeypatch.setattr(output, 'get_setting', setting)
    destination = run(output.title_edit_destination(None, title, media))
    root = tmp_path / ('movie-edits' if kind == MediaType.movie else 'tv-edits')
    expected = Path(relative).with_name(Path(relative).stem + ' - Vulgarr Edit.mkv')
    assert destination == root / expected
    assert not destination.exists()  # Resolution itself never writes anything.


def test_blank_output_preserves_legacy_behavior(tmp_path):
    assert output.edit_destination(tmp_path/'movie.mkv', tmp_path, '  ') is None


def test_relative_output_rejected(tmp_path):
    with pytest.raises(ValueError, match='absolute'):
        output.edit_destination(tmp_path/'movie.mkv', tmp_path, 'relative/path')


def test_outside_media_root_uses_collision_safe_external_tree(tmp_path):
    source = tmp_path / 'second-mount' / 'Show' / 'episode.mkv'
    result = output.edit_destination(source, tmp_path/'media', str(tmp_path/'edits'))
    assert result.is_relative_to(tmp_path/'edits'/'_external')
    assert result.parts[-3:] == ('second-mount', 'Show', 'episode - Vulgarr Edit.mkv')


def test_duplicate_basenames_do_not_collide(tmp_path):
    a = output.edit_destination(tmp_path/'A'/'episode.mkv', tmp_path, str(tmp_path/'edits'))
    b = output.edit_destination(tmp_path/'B'/'episode.mkv', tmp_path, str(tmp_path/'edits'))
    assert a != b


def test_publish_atomic_same_filesystem(tmp_path):
    staged, dest = tmp_path/'staged', tmp_path/'subdir'/'edited.mkv'
    staged.write_bytes(b'new')
    run(output.publish_file(staged, dest))
    assert dest.read_bytes() == b'new'
    assert not staged.exists()


def test_cross_device_publish_stages_on_destination(monkeypatch, tmp_path):
    staged, dest = tmp_path/'staged', tmp_path/'edits'/'movie.mkv'
    staged.write_bytes(b'new')
    dest.parent.mkdir()
    dest.write_bytes(b'old')
    replace = Path.replace
    def cross_device(self, target):
        if self == staged:
            raise OSError(errno.EXDEV, 'Different filesystems')
        assert self.parent == dest.parent
        assert dest.read_bytes() == b'old'
        return replace(self, target)
    monkeypatch.setattr(Path, 'replace', cross_device)
    run(output.publish_file(staged, dest))
    assert dest.read_bytes() == b'new'
    assert not staged.exists()
    assert list(dest.parent.iterdir()) == [dest]


def test_cross_device_copy_failure_preserves_previous_edit(monkeypatch, tmp_path):
    staged, dest = tmp_path/'staged', tmp_path/'edited.mkv'
    staged.write_bytes(b'new')
    dest.write_bytes(b'old')
    replace = Path.replace
    def cross_device(self, target):
        if self == staged:
            raise OSError(errno.EXDEV, 'Different filesystems')
        return replace(self, target)
    def failed_copy(source, target, **kwargs):
        target.write(b'partial')
        raise OSError('Disk full')
    monkeypatch.setattr(Path, 'replace', cross_device)
    monkeypatch.setattr(output.shutil, 'copyfileobj', failed_copy)
    with pytest.raises(OSError, match='Disk full'):
        run(output.publish_file(staged, dest))
    assert dest.read_bytes() == b'old'
    assert staged.read_bytes() == b'new'
    assert not list(tmp_path.glob('.*.spf-tmp.*'))


def test_output_bookkeeping_migrates_existing_database_idempotently(monkeypatch):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db import session
    names = {'clean_audio_path', 'clean_audio_source_fingerprint', 'source_clean_track_audio_indices'}
    additions = [column for column in session._COLUMNS_ADDED_LATER if column[1] in names]
    assert len(additions) == 3
    monkeypatch.setattr(session, '_COLUMNS_ADDED_LATER', additions)
    async def check():
        engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        try:
            async with engine.begin() as connection:
                await connection.execute(text('CREATE TABLE titles (id INTEGER PRIMARY KEY)'))
                await connection.execute(text('INSERT INTO titles VALUES (1)'))
                await session._run_column_migrations(connection)
                await session._run_column_migrations(connection)
                columns = (await connection.execute(text('PRAGMA table_info(titles)'))).fetchall()
                assert names <= {row[1] for row in columns}
                row = (await connection.execute(text('SELECT clean_audio_path, clean_audio_source_fingerprint, source_clean_track_audio_indices FROM titles'))).one()
                assert tuple(row) == (None, None, None)
        finally:
            await engine.dispose()
    run(check())


def test_orphan_cleanup_retains_active_master_and_unrelated_files(tmp_path):
    from app.external_edits import _sweep_stale_work
    source = tmp_path/'source'/'movie.mkv'
    source.parent.mkdir()
    source.write_bytes(b'original')
    destination = tmp_path/'edits'/'movie - Vulgarr Edit.mkv'
    destination.parent.mkdir()
    destination.write_bytes(b'visible edit')
    abandoned = destination.parent/'.movie.spf-tmp.dead.mkv'
    abandoned.write_bytes(b'partial')
    import os
    os.utime(abandoned, (1, 1))
    unrelated = destination.parent/'.other.spf-tmp.dead.mkv'
    unrelated.write_bytes(b'unrelated')
    directory = destination.parent/'.vulgarr'/'clean_audio'/'1'
    directory.mkdir(parents=True)
    retained, orphan = directory/'active.mka', directory/'orphan.mka'
    retained.write_bytes(b'active')
    orphan.write_bytes(b'orphan')
    _sweep_stale_work(source, destination, directory, str(retained))
    assert retained.read_bytes() == b'active'
    assert not orphan.exists() and not abandoned.exists()
    assert unrelated.read_bytes() == b'unrelated'
    assert source.read_bytes() == b'original'
    assert destination.read_bytes() == b'visible edit'


def test_fingerprint_detects_source_replacement(tmp_path):
    source = tmp_path/'video'
    source.write_bytes(b'old')
    before = output.source_fingerprint(source)
    source.write_bytes(b'new source')
    assert output.source_fingerprint(source) != before


def test_audio_remux_external_never_backs_up_or_replaces_source(monkeypatch, tmp_path):
    from app.mux import remux
    video = tmp_path/'media'/'movie.mkv'
    video.parent.mkdir()
    video.write_bytes(b'original')
    dest = tmp_path/'separate'/'movie - Vulgarr Edit.mkv'
    source_probe = {'format': {'duration': '10'}, 'streams': [
        {'codec_type': 'video', 'index': 0},
        {'codec_type': 'audio', 'index': 1, 'tags': {}, 'disposition': {'default': 1}},
    ]}
    async def probe(*args):
        return source_probe
    async def build(**kwargs):
        assert kwargs['video_path'] == video
        assert kwargs['staging_dir'] == dest.parent
        dest.parent.mkdir()
        staged = dest.parent/'staged.mkv'
        staged.write_bytes(b'cleaned')
        return staged
    async def verify(**kwargs):
        assert kwargs['tmp_path'].read_bytes() == b'cleaned'
    def forbidden(*args):
        pytest.fail('External processing must never move or back up the source')
    monkeypatch.setattr(remux, 'probe', probe)
    monkeypatch.setattr(remux, 'build_clean_track', build)
    monkeypatch.setattr(remux, 'verify_output', verify)
    monkeypatch.setattr(remux, 'backup_path_for', forbidden)
    result = run(remux.remux_with_clean_track(video_path=video,
        intervals_by_level=[(Severity.child, [])], matches=[], media_root=video.parent,
        backup_root=tmp_path/'backups', ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe',
        clean_track_title='Clean', clean_track_language='eng', output_path=dest))
    assert result.output_path == dest
    assert result.backup_path is None
    assert dest.read_bytes() == b'cleaned'
    assert video.read_bytes() == b'original'
    assert not (tmp_path/'backups').exists()


def test_final_mux_uses_cached_audio_and_original_metadata(monkeypatch, tmp_path):
    from app.mux import scene_blur
    commands = []
    async def command(cmd, **kwargs):
        commands.append(cmd)
        return 0, '', ''
    monkeypatch.setattr(scene_blur, '_run', command)
    from app.mux.remux import _audio_streams
    real_audio = _audio_streams({'streams': [{'codec_type': 'audio'}, {'codec_type': 'audio'},
        {'codec_type': 'audio', 'tags': {'VULGARR_TIMING_ANCHOR': '1'}}]})
    assert len(real_audio) == 2
    run(scene_blur._mux_final_output(ffmpeg_bin='ffmpeg', video_path=tmp_path/'original.mkv',
        concatenated_video_path=tmp_path/'blurred.mkv', concatenated_audio_path=None,
        audio_source_path=tmp_path/'clean.mka', audio_streams=real_audio,
        out_path=tmp_path/'output.mkv', timeout=30))
    cmd = commands[0]
    assert [cmd[i+1] for i, item in enumerate(cmd) if item == '-i'] == [
        str(tmp_path/'blurred.mkv'), str(tmp_path/'original.mkv'), str(tmp_path/'clean.mka')]
    assert '2:a:0' in cmd and '2:a:1' in cmd
    assert '2:a:2' not in cmd  # The timing-only anchor must never leave the cache.
    assert '1:s?' in cmd and cmd[cmd.index('-map_metadata')+1] == '1'


class FakeSession:
    def __init__(self, approved=None):
        self.approved = approved
        self.commits = 0
    async def scalar(self, query):
        return self.approved
    async def execute(self, query):
        return SimpleNamespace(scalars=lambda: [])
    async def commit(self):
        self.commits += 1


@pytest.mark.parametrize('approved', [None, 1])
def test_external_audio_composes_approved_blurs_without_modifying_source(monkeypatch, tmp_path, approved):
    from app import external_edits
    source = tmp_path/'media'/'movie.mkv'
    source.parent.mkdir()
    source.write_bytes(b'original')
    dest = tmp_path/'edits'/'movie - Vulgarr Edit.mkv'
    old_cache = tmp_path/'old.mka'
    old_cache.write_bytes(b'old cache')
    title = SimpleNamespace(id=7, video_path=str(source), clean_audio_path=str(old_cache),
                            clean_audio_source_fingerprint='previous')
    monkeypatch.setattr(external_edits, 'app_settings', SimpleNamespace(data_dir=tmp_path/'data'))
    outcome = SimpleNamespace(matched_cue_count=3, clean_track_indices=[1])
    async def process(session, **options):
        assert options['known_clean_indices'] is None
        assert options['video_path'] == source
        path = options['output_path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'clean audio plus original video')
        return outcome
    async def cache(video, master):
        master.parent.mkdir(parents=True)
        master.write_bytes(b'clean master')
    async def blur(session, current_title, job_id, **options):
        assert current_title is title
        assert job_id.startswith('audio-')
        assert options['output_path'] == dest
        assert options['commit'] is False
        assert options['audio_source_path'].read_bytes() == b'clean master'
        assert title.clean_audio_path == str(old_cache)  # Progress commits must still see the old master.
        dest.write_bytes(b'clean audio plus blurred original video')
    monkeypatch.setattr(external_edits, 'process_video', process)
    monkeypatch.setattr(external_edits, '_cache_audio', cache)
    monkeypatch.setattr(external_edits, 'apply_scene_blur', blur)
    session = FakeSession(approved)
    result = run(external_edits.process_external_video(session, title=title, job_id=2,
        destination=dest, title_id=7, video_path=source, known_clean_indices=[99]))
    assert result is outcome
    assert source.read_bytes() == b'original'
    assert b'clean audio' in dest.read_bytes()
    assert (b'blurred' in dest.read_bytes()) == (approved is not None)
    assert Path(title.clean_audio_path).read_bytes() == b'clean master'
    assert Path(title.clean_audio_path).is_relative_to(dest.parent/'.vulgarr')
    assert not old_cache.exists()
    assert not list(dest.parent.glob('.*.spf-tmp.*'))
    assert session.commits == 1


def test_combined_failure_keeps_previous_edit_and_audio_master(monkeypatch, tmp_path):
    from app import external_edits
    source, dest, old_cache = tmp_path/'source.mkv', tmp_path/'edited.mkv', tmp_path/'old.mka'
    source.write_bytes(b'original')
    dest.write_bytes(b'previous combined edit')
    old_cache.write_bytes(b'old audio')
    title = SimpleNamespace(id=1, video_path=str(source), clean_audio_path=str(old_cache),
                            clean_audio_source_fingerprint='old fingerprint')
    monkeypatch.setattr(external_edits, 'app_settings', SimpleNamespace(data_dir=tmp_path/'data'))
    async def process(session, **options):
        options['output_path'].write_bytes(b'new unblurred audio output')
        return SimpleNamespace()
    async def cache(video, path):
        path.parent.mkdir(parents=True)
        path.write_bytes(b'new clean master')
    async def fail(*args, **kwargs):
        raise RemuxError('Blur failed')
    monkeypatch.setattr(external_edits, 'process_video', process)
    monkeypatch.setattr(external_edits, '_cache_audio', cache)
    monkeypatch.setattr(external_edits, 'apply_scene_blur', fail)
    with pytest.raises(RemuxError, match='Blur failed'):
        run(external_edits.process_external_video(FakeSession(1), title=title, job_id=3,
            destination=dest, title_id=1, video_path=source))
    assert source.read_bytes() == b'original'
    assert dest.read_bytes() == b'previous combined edit'
    assert title.clean_audio_path == str(old_cache)
    assert old_cache.read_bytes() == b'old audio'
    assert not list((dest.parent/'.vulgarr').rglob('*.mka'))


def test_settings_save_export_and_validation(monkeypatch, tmp_path):
    import httpx
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from app.routers import settings
    from app.db.session import DEFAULT_SETTINGS
    stored = {}
    session = FakeSession()
    @asynccontextmanager
    async def get_session():
        yield session
    async def get_setting(session, key):
        return stored.get(key, DEFAULT_SETTINGS.get(key))
    async def set_setting(session, key, value):
        stored[key] = value
    monkeypatch.setattr(settings, 'get_session', get_session)
    monkeypatch.setattr(settings, 'get_setting', get_setting)
    monkeypatch.setattr(settings, 'set_setting', set_setting)
    app = FastAPI()
    app.include_router(settings.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            form = {'concurrency_cap': '2', 'off_hours_start': '01:00', 'off_hours_end': '07:00',
                    'movie_output_folder': str(tmp_path/'movies'), 'tv_output_folder': str(tmp_path/'tv')}
            response = await client.post('/settings', data=form)
            assert response.status_code == 200
            assert stored['movie_output_folder'] == form['movie_output_folder']
            assert stored['tv_output_folder'] == form['tv_output_folder']
            assert 'Movie output folder' in response.text and 'TV output folder' in response.text
            exported = (await client.get('/settings/export')).json()
            assert exported['movie_output_folder'] == form['movie_output_folder']
            before = dict(stored)
            form['tv_output_folder'] = 'relative/path'
            response = await client.post('/settings', data=form)
            assert 'No settings were saved' in response.text
            assert stored == before
            form['movie_output_folder'] = form['tv_output_folder'] = ''
            await client.post('/settings', data=form)
            assert stored['movie_output_folder'] == stored['tv_output_folder'] == ''
    run(check())


@pytest.mark.parametrize('cache_state', ['valid', 'missing', 'changed_source', 'legacy'])
def test_scene_apply_preserves_clean_audio_and_rejects_stale_master(monkeypatch, tmp_path, cache_state):
    from app.scenes import pipeline
    from app.db.session import DEFAULT_SETTINGS
    from app.db.models import SceneReviewStatus
    source = tmp_path/'media'/'Movies'/'movie.mkv'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original video')
    cache = tmp_path/'cache.mka'
    cache.write_bytes(b'clean audio with no scene mutes')
    destination_root = tmp_path/'edits'
    title = SimpleNamespace(id=1, media_type=MediaType.movie, video_path=str(source),
        clean_audio_path=str(cache), clean_audio_source_fingerprint=output.source_fingerprint(source))
    if cache_state == 'missing':
        cache.unlink()
    if cache_state == 'changed_source':
        source.write_bytes(b'new source edition')
    scene = SimpleNamespace(start_seconds=2, end_seconds=3, mute_audio=True,
                            status=SceneReviewStatus.approved, applied_at=None)
    class Session(FakeSession):
        async def execute(self, query):
            self.queries = getattr(self, 'queries', 0) + 1
            scenes = [scene] if self.queries == 1 else []
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: scenes))
    calls = []
    async def setting(session, key):
        if key in {'movie_output_folder', 'tv_output_folder'}:
            return '' if cache_state == 'legacy' else str(destination_root)
        return DEFAULT_SETTINGS[key]
    async def apply(**options):
        calls.append(options)
        return options.get('output_path', source.with_name('movie - Vulgarr Edit.mkv'))
    monkeypatch.setattr(pipeline, 'get_setting', setting)
    monkeypatch.setattr(output, 'get_setting', setting)
    monkeypatch.setattr(pipeline, 'apply_blur', apply)
    monkeypatch.setattr(pipeline, 'app_settings', SimpleNamespace(data_dir=tmp_path/'data',
        media_root=tmp_path/'media', ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe'))
    if cache_state in {'missing', 'changed_source'}:
        with pytest.raises(ValueError, match='Reprocess audio'):
            run(pipeline.apply_scene_blur(Session(), title, 42))
        assert not calls
    else:
        run(pipeline.apply_scene_blur(Session(), title, 42))
        assert calls[0]['video_path'] == source
        if cache_state == 'valid':
            assert calls[0]['audio_source_path'] == cache
            assert calls[0]['output_path'].is_relative_to(destination_root)
        else:
            assert 'audio_source_path' not in calls[0]
            assert 'output_path' not in calls[0]
    if cache.exists():
        assert cache.read_bytes() == b'clean audio with no scene mutes'


def test_apply_blur_verification_counts_cached_clean_tracks(monkeypatch, tmp_path):
    from app.mux import scene_blur
    from app.audio.mute import MuteInterval
    source = tmp_path/'source.mkv'
    source.write_bytes(b'original')
    cache = tmp_path/'clean.mka'
    cache.write_bytes(b'audio')
    dest = tmp_path/'library'/'edited.mkv'
    seen = []
    async def probe(binary, path):
        streams = [{'codec_type': 'audio'}, {'codec_type': 'audio'}] if path == cache else [
            {'codec_type': 'video'}, {'codec_type': 'audio'}]
        return {'streams': streams, 'format': {'duration': '10'}}
    async def build(**options):
        assert options['audio_source_path'] == cache
        assert options['video_path'] == source
        assert options['staging_dir'] == dest.parent
        dest.parent.mkdir(exist_ok=True)
        path = dest.parent/'staged.mkv'
        path.write_bytes(b'combined edit')
        return path
    async def verify(**options):
        seen.append(options)
        assert len(options['original_probe']['streams']) == 3
    monkeypatch.setattr(scene_blur, 'probe', probe)
    monkeypatch.setattr(scene_blur, 'build_blurred_video', build)
    monkeypatch.setattr(scene_blur, 'verify_blurred_output', verify)
    result = run(scene_blur.apply_blur(video_path=source, blur_intervals=[MuteInterval(2, 3)],
        mute_intervals=[], ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe', video_crf=23,
        video_preset='fast', work_dir=tmp_path/'work', audio_source_path=cache, output_path=dest))
    assert result == dest
    assert dest.read_bytes() == b'combined edit'
    assert source.read_bytes() == b'original'
    assert len(seen) == 1


def test_scene_mute_pipeline_reads_cached_audio_not_original_tracks(monkeypatch, tmp_path):
    from app.mux import scene_blur
    from app.audio.mute import MuteInterval
    source = tmp_path/'media'/'movie.mkv'
    source.parent.mkdir()
    source.write_bytes(b'original')
    cache = tmp_path/'master.mka'
    cache.write_bytes(b'clean audio')
    calls = []
    async def probe(binary, path):
        streams = ([{'codec_type': 'audio'}, {'codec_type': 'audio'},
            {'codec_type': 'audio', 'tags': {'VULGARR_TIMING_ANCHOR': '1'}}] if path == cache else
            [{'codec_type': 'video', 'codec_name': 'h264'}, {'codec_type': 'audio'}])
        return {'streams': streams, 'format': {'duration': '10'}}
    async def video(*args, **options):
        assert args[1] == source
        args[8].write_bytes(b'blurred')
    async def split(binary, path, boundaries, count, directory):
        assert path == cache and count == 2
        calls.append('split')
    async def mute(binary, path, segment, count, bitrate, output_path):
        assert path == cache and count == 2
        output_path.write_bytes(b'muted')
        calls.append('mute')
    async def normalize(binary, path, output_path):
        output_path.write_bytes(b'copy')
    async def concat(binary, paths, count, output_path):
        assert count == 2
        output_path.write_bytes(b'processed audio')
    async def mux(**options):
        assert options['video_path'] == source
        assert options['audio_source_path'] == cache
        assert options['concatenated_audio_path'].read_bytes() == b'processed audio'
        options['out_path'].write_bytes(b'finished edit')
    for name, replacement in [('probe', probe), ('_build_video_whole_file_reencode', video),
            ('_split_copy_audio_segments', split), ('_extract_mute_audio_segment', mute),
            ('_normalize_segment_timestamps', normalize), ('_concat_audio_segments', concat),
            ('_mux_final_output', mux)]:
        monkeypatch.setattr(scene_blur, name, replacement)
    staged = run(scene_blur.build_blurred_video(video_path=source,
        blur_intervals=[MuteInterval(2, 3)], mute_intervals=[MuteInterval(2, 3)],
        ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe', video_crf=23, video_preset='fast',
        work_dir=tmp_path/'work', audio_source_path=cache, staging_dir=tmp_path/'out',
        force_whole_file_video=True))
    assert staged.parent == tmp_path/'out'
    assert staged.read_bytes() == b'finished edit'
    assert calls == ['split', 'mute']
    assert list(source.parent.iterdir()) == [source]


def test_source_track_indices_survive_repeated_external_runs(monkeypatch, tmp_path):
    from app import external_edits
    source, dest = tmp_path/'original.mp4', tmp_path/'edited.mp4'
    source.write_bytes(b'original with an older tagless clean track')
    title = SimpleNamespace(id=1, video_path=str(source), clean_audio_path=None,
        clean_audio_source_fingerprint=None, source_clean_track_audio_indices=None)
    monkeypatch.setattr(external_edits, 'app_settings', SimpleNamespace(data_dir=tmp_path/'data'))
    seen = []
    async def process(session, **options):
        seen.append(options['known_clean_indices'])
        options['output_path'].write_bytes(b'edit')
        return SimpleNamespace()
    async def cache(video, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'clean audio')
    monkeypatch.setattr(external_edits, 'process_video', process)
    monkeypatch.setattr(external_edits, '_cache_audio', cache)
    run(external_edits.process_external_video(FakeSession(), title=title, job_id=1,
        destination=dest, title_id=1, video_path=source, known_clean_indices=[1]))
    run(external_edits.process_external_video(FakeSession(), title=title, job_id=2,
        destination=dest, title_id=1, video_path=source, known_clean_indices=[99]))
    assert seen == [[1], [1]]  # Never interpret destination index 99 as source audio.
    assert source.read_bytes() == b'original with an older tagless clean track'


@pytest.mark.parametrize('mode', ['valid_shorter_audio', 'delayed_audio', 'shifted_audio', 'wrong_codec', 'missing_track', 'truncated'])
def test_audio_master_validation_and_silent_video_tail(monkeypatch, tmp_path, mode):
    from app import external_edits
    video, cache = tmp_path/'video.mkv', tmp_path/'master.mka'
    video.write_bytes(b'video')
    cache.write_bytes(b'previous master')
    audio = {'codec_type': 'audio', 'codec_name': 'aac', 'sample_rate': '48000',
             'channels': 2, 'duration': '90', 'start_time': '5' if mode in {'delayed_audio', 'shifted_audio'} else '0'}
    async def command(cmd, **kwargs):
        assert '-c:a' in cmd and cmd[cmd.index('-c:a')+1] == 'copy'
        Path(cmd[-1]).write_bytes(b'new master')
        return 0, '', ''
    async def probe(binary, path):
        if path == video:
            return {'streams': [{'codec_type': 'video'}, audio], 'format': {'duration': '100'}}
        copied = dict(audio)
        if mode == 'wrong_codec':
            copied['codec_name'] = 'mp3'
        if mode == 'truncated':
            copied['duration'] = '70'
        if mode == 'shifted_audio':
            copied['start_time'] = '0'
        anchor = {'codec_type': 'audio', 'tags': {'VULGARR_TIMING_ANCHOR': '1'}}
        return {'streams': [anchor] if mode == 'missing_track' else [copied, anchor],
                'format': {'duration': '90', 'start_time': '0'}}
    monkeypatch.setattr(external_edits, '_run', command)
    monkeypatch.setattr(external_edits, 'probe', probe)
    if mode in {'valid_shorter_audio', 'delayed_audio'}:
        run(external_edits._cache_audio(video, cache))
        assert cache.read_bytes() == b'new master'
    else:
        with pytest.raises(RemuxError):
            run(external_edits._cache_audio(video, cache))
        assert cache.read_bytes() == b'previous master'
    assert not cache.with_name(cache.stem + '.tmp.mka').exists()


def test_cancelled_cross_device_publish_finishes_without_partial_file(monkeypatch, tmp_path):
    import threading
    staged, destination = tmp_path/'staged', tmp_path/'edit.mkv'
    staged.write_bytes(b'complete new edit')
    destination.write_bytes(b'previous edit')
    entered, release = threading.Event(), threading.Event()
    replace = Path.replace
    def cross_device(self, target):
        if self == staged:
            raise OSError(errno.EXDEV, 'Different filesystems')
        return replace(self, target)
    def copy(source, target, **kwargs):
        entered.set()
        assert release.wait(2)
        target.write(source.read())
    monkeypatch.setattr(Path, 'replace', cross_device)
    monkeypatch.setattr(output.shutil, 'copyfileobj', copy)
    async def check():
        task = asyncio.create_task(output.publish_file(staged, destination))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(check())
    assert destination.read_bytes() == b'complete new edit'
    assert not staged.exists()
    assert not list(tmp_path.glob('.*.spf-tmp.*'))
