"""Whole-file recovery must query workers even if no segment encode was reached."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI

from app.mux import remote_render
from app.vision import remote_source
from remote_worker import render


def run(coro):
    return asyncio.run(coro)


def payload():
    return dict(source_url='http://vulgarr:8000/internal/source?signature=private',
        source_codec='hevc', duration=10, width=1920, height=1080, pix_fmt='yuv420p',
        crf=23, preset='ultrafast', filter_complex="[0:v]boxblur=luma_radius=90:luma_power=8:chroma_radius=90:chroma_power=8:enable='between(t,2.000,3.000)'[vb]")


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setenv('SPF_SOURCE_BASE_URL', 'http://vulgarr:8000')
    monkeypatch.setenv('SPF_RENDER_WORK_DIR', str(tmp_path/'renders'))
    monkeypatch.setattr(render, '_jobs', {})
    monkeypatch.setattr(render, '_completed', 0)
    monkeypatch.setattr(render, '_latency', 0.0)
    monkeypatch.setattr(render, 'encode_slot', asyncio.Semaphore(1))
    yield render
    for job in render._jobs.values():
        job['directory'].cleanup()


@pytest.mark.parametrize('gpu_available', [True, False])
def test_worker_job_renders_remotely_and_download_cleans_artifacts(worker, monkeypatch, gpu_available):
    commands = []
    async def ffmpeg(command, job):
        commands.append(command)
        if not gpu_available and 'hevc_nvenc' in command:
            return 1
        Path(command[-1]).write_bytes(b'worker-rendered-video')
        job['progress'] = .5
        return 0
    async def valid(*args):
        return True
    monkeypatch.setattr(worker, '_run_ffmpeg', ffmpeg)
    monkeypatch.setattr(worker, '_valid_output', valid)
    app = FastAPI()
    app.include_router(worker.router)
    async def check():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as client:
                response = await client.post('/render', json=payload())
                assert response.status_code == 202
                identity = response.json()['id']
                for _ in range(20):
                    await asyncio.sleep(0)
                    status = await client.get('/render/'+identity)
                    if status.json()['state'] == 'done':
                        break
                assert status.json()['state'] == 'done'
                assert status.json()['backend'] == ('NVENC' if gpu_available else 'worker CPU')
                assert 'private' not in status.text and 'source_url' not in status.text
                if not gpu_available:
                    assert len(commands) == 3
                    assert 'libx265' in commands[-1]
                result = await client.get('/render/'+identity+'/output')
                assert result.content == b'worker-rendered-video'
                assert (await client.get('/render/'+identity)).status_code == 404
                assert worker.render_stats()['render_requests_served'] == 1
                assert not worker._jobs
        finally:
            await worker.shutdown_renders()
    run(check())


def test_worker_queue_limits_source_restrictions_and_cancellation(worker, monkeypatch):
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def blocked(command, job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    monkeypatch.setattr(worker, '_run_ffmpeg', blocked)
    app = FastAPI()
    app.include_router(worker.router)
    async def check():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as client:
                bad = payload()
                bad['source_url'] = 'http://untrusted/internal/source'
                assert (await client.post('/render', json=bad)).status_code == 422
                bad = payload()
                bad['filter_complex'] = 'movie=/etc/passwd[vb]'
                assert (await client.post('/render', json=bad)).status_code == 422
                first = (await client.post('/render', json=payload())).json()['id']
                await entered.wait()
                second = (await client.post('/render', json=payload())).json()['id']
                assert (await client.post('/render', json=payload())).status_code == 429
                assert (await client.get('/render/'+first+'/output')).status_code == 409
                assert (await client.delete('/render/'+first)).status_code == 200
                assert stopped.is_set()
                assert (await client.get('/render/'+first)).status_code == 404
                await client.delete('/render/'+second)
                assert not worker._jobs
        finally:
            await worker.shutdown_renders()
    run(check())


def test_download_send_error_releases_registry_and_artifacts(worker, tmp_path):
    import tempfile
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    output = Path(directory.name)/'video.mkv'
    output.write_bytes(b'video')
    identity = 'a'*32
    job = {'output': output, 'directory': directory}
    worker._jobs[identity] = job
    response = worker.RenderFileResponse(identity, job)
    async def disconnected(message): raise ConnectionError('Client disconnected')
    async def receive(): return {'type': 'http.disconnect'}
    async def check():
        with pytest.raises(ConnectionError):
            await response({'type': 'http', 'method': 'GET', 'headers': []}, receive, disconnected)
    run(check())
    assert identity not in worker._jobs
    assert not Path(directory.name).exists()


@pytest.mark.parametrize('change', [None, 'codec', 'dimensions', 'pixel_format', 'duration', 'malformed'])
def test_worker_validates_whole_video_metadata(worker, monkeypatch, tmp_path, change):
    output = tmp_path/'video.mkv'
    output.write_bytes(b'video')
    data = {'streams': [{'codec_type': 'video', 'codec_name': 'hevc', 'width': 1920,
                        'height': 1080, 'pix_fmt': 'yuv420p'}], 'format': {'duration': '10'}}
    if change == 'codec': data['streams'][0]['codec_name'] = 'h264'
    if change == 'dimensions': data['streams'][0]['width'] = 1280
    if change == 'pixel_format': data['streams'][0]['pix_fmt'] = 'yuv420p10le'
    if change == 'duration': data['format']['duration'] = '2'
    class Process:
        returncode = 0
        async def communicate(self):
            return (b'invalid' if change == 'malformed' else json.dumps(data).encode()), None
    async def spawn(*args, **kwargs): return Process()
    monkeypatch.setattr(worker.asyncio, 'create_subprocess_exec', spawn)
    assert run(worker._valid_output(output, worker.RenderRequest(**payload()))) == (change is None)


def test_cancel_kills_actual_worker_ffmpeg(worker, monkeypatch):
    entered, killed = asyncio.Event(), asyncio.Event()
    class Reader:
        def __aiter__(self): return self
        async def __anext__(self):
            await killed.wait()
            raise StopAsyncIteration
    class Process:
        returncode = None
        stdout = Reader()
        async def wait(self):
            await killed.wait()
            return self.returncode
        def kill(self):
            self.returncode = -9
            killed.set()
    async def spawn(*args, **kwargs):
        entered.set()
        return Process()
    monkeypatch.setattr(worker.asyncio, 'create_subprocess_exec', spawn)
    async def check():
        task = asyncio.create_task(worker._run_ffmpeg(['ffmpeg'], {'params': SimpleNamespace(duration=10)}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert killed.is_set()
    run(check())


def configure_client(monkeypatch):
    monkeypatch.setenv('SPF_REMOTE_ENCODE_URL', 'http://worker')
    monkeypatch.setenv('SPF_INTERNAL_BASE_URL', 'http://vulgarr:8000')
    monkeypatch.setenv('SPF_INTERNAL_TOKEN', 'test-secret')


def test_worker_startup_reclaims_only_its_dedicated_render_artifacts(worker, tmp_path):
    root = tmp_path/'renders'
    orphan = root/'vulgarr-render-abandoned'
    orphan.mkdir(parents=True)
    (orphan/'video.mkv').write_bytes(b'abandoned')
    unrelated = root/'unrelated'
    unrelated.mkdir()
    (unrelated/'keep').write_bytes(b'keep')
    run(worker.startup_renders())
    assert not orphan.exists()
    assert (unrelated/'keep').read_bytes() == b'keep'


def test_worker_panel_includes_encoder_without_mutating_classifier_pool(monkeypatch):
    from app.routers import remote_workers
    classifiers = ['http://classifier']
    monkeypatch.setattr(remote_workers, '_classifier_urls', lambda: classifiers)
    monkeypatch.setenv('SPF_REMOTE_ENCODE_URL', 'http://encoder/')
    queries = []
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url):
            queries.append(url)
            return httpx.Response(200, json={'whole_video_render': True}, request=httpx.Request('GET', url))
    monkeypatch.setattr(remote_workers.httpx, 'AsyncClient', lambda **kwargs: Client())
    monkeypatch.setattr(remote_workers.templates, 'TemplateResponse', lambda name, context: context)
    result = run(remote_workers.workers(None))
    assert {row['url'] for row in result['workers']} == {'http://classifier', 'http://encoder'}
    assert classifiers == ['http://classifier']
    assert 'http://encoder/stats' in queries


def test_render_source_capability_covers_job_deadline(monkeypatch, tmp_path):
    configure_client(monkeypatch)
    monkeypatch.setattr(remote_source.time, 'time', lambda: 100)
    url = remote_source.source_url(tmp_path/'video', ttl_seconds=90000)
    query = parse_qs(urlsplit(url).query)
    assert query['expires'] == ['90100']
    assert query['signature'][0] == remote_source.source_signature(query['path'][0], 90100)


@pytest.mark.parametrize('failure', [None, 'worker', 'validation', 'bad_progress'])
def test_host_polls_downloads_validates_and_keeps_previous_output_on_failure(monkeypatch, tmp_path, failure):
    configure_client(monkeypatch)
    monkeypatch.setenv('SPF_REMOTE_ENCODE_URL', 'http://user:worker-secret@worker')
    identity = 'a'*32
    requests, stages = [], []
    out = tmp_path/'video.mkv'
    out.write_bytes(b'previous')
    async def handle(request):
        requests.append((request.method, request.url.path))
        if request.method == 'DELETE': return httpx.Response(200, json={'ok': True})
        if request.method == 'POST': return httpx.Response(202, json={'id': identity})
        if request.url.path.endswith('/output'): return httpx.Response(200, content=b'worker-video')
        return httpx.Response(200, json={'state': 'failed' if failure == 'worker' else 'done',
            'progress': 'NaN' if failure == 'bad_progress' else 1, 'backend': 'NVENC'})
    async def validate(path):
        assert path.read_bytes() == b'worker-video'
        if failure == 'validation': raise ValueError('Bad decode')
    async def stage(message): stages.append(message)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(remote_render.httpx, 'AsyncClient', lambda **kwargs: client)
    result = run(remote_render.try_remote_render(video_path=tmp_path/'original.mkv', out_path=out,
        source_codec='hevc', source_video={'width': 1920, 'height': 1080, 'pix_fmt': 'yuv420p'},
        duration=10, crf=23, preset='ultrafast', filter_complex='[0:v]null[vb]',
        timeout=10, validate=validate, on_stage=stage))
    assert result == (failure is None)
    assert out.read_bytes() == (b'worker-video' if failure is None else b'previous')
    assert ('POST', '/render') in requests and ('DELETE', '/render/'+identity) in requests
    assert not out.with_name(out.name+'.remote.tmp').exists()
    assert any('whole-file' in message for message in stages)
    assert all('worker-secret' not in message for message in stages)


def test_missing_source_transport_is_explained_without_silent_local_selection(monkeypatch, tmp_path):
    configure_client(monkeypatch)
    monkeypatch.delenv('SPF_INTERNAL_TOKEN')
    stages = []
    def forbidden(**kwargs): pytest.fail('Cannot submit without source transport')
    async def stage(message): stages.append(message)
    async def validate(path): pytest.fail('No output to validate')
    monkeypatch.setattr(remote_render.httpx, 'AsyncClient', forbidden)
    assert not run(remote_render.try_remote_render(video_path=tmp_path/'original', out_path=tmp_path/'out',
        source_codec='hevc', source_video={}, duration=10, crf=23, preset='fast',
        filter_complex='[0:v]null[vb]', timeout=10, validate=validate, on_stage=stage))
    assert 'SPF_INTERNAL_TOKEN' in stages[0] and 'server CPU' in stages[0]


@pytest.mark.parametrize('codec,encoder', [('hevc', 'libx265'), ('h264', 'libx264'),
    ('vp9', 'libvpx-vp9'), ('av1', 'libsvtav1'), ('mpeg2video', 'mpeg2video'), ('mpeg4', 'mpeg4')])
def test_worker_software_codec_and_options_match_source(codec, encoder, tmp_path):
    values = payload()
    values['source_codec'] = codec
    command = render._command(render.RenderRequest(**values), tmp_path/'out', 'worker CPU', False)
    assert command[command.index('-c:v')+1] == encoder
    if codec in {'av1', 'vp9', 'mpeg2video', 'mpeg4'}:
        assert 'ultrafast' not in command  # Not a valid native/SVT/VPx preset.


def test_host_cancellation_cancels_worker_job(monkeypatch, tmp_path):
    configure_client(monkeypatch)
    started = asyncio.Event()
    deleted = []
    identity = 'a'*32
    async def handle(request):
        if request.method == 'DELETE':
            deleted.append(request.url.path)
            return httpx.Response(200)
        if request.method == 'POST': return httpx.Response(202, json={'id': identity})
        started.set()
        await asyncio.Event().wait()
    async def validate(path): pass
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(remote_render.httpx, 'AsyncClient', lambda **kwargs: client)
    async def check():
        task = asyncio.create_task(remote_render.try_remote_render(video_path=tmp_path/'original',
            out_path=tmp_path/'out', source_codec='hevc', source_video=payload(), duration=10,
            crf=23, preset='fast', filter_complex='[0:v]null[vb]', timeout=10, validate=validate))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    run(check())
    assert deleted == ['/render/'+identity]


@pytest.mark.parametrize('remote_ok', [True, False])
def test_whole_file_entrypoint_queries_worker_before_any_local_encode(monkeypatch, tmp_path, remote_ok):
    from app.mux import scene_blur
    from app.audio.mute import MuteInterval
    configure_client(monkeypatch)
    calls = []
    async def remote(**options):
        calls.append('remote')
        assert 'between(t,2.000,3.000)' in options['filter_complex']
        if remote_ok: options['out_path'].write_bytes(b'remote')
        return remote_ok
    async def local(command, **options):
        calls.append('server CPU')
        assert 'libx265' in command
        Path(command[-1]).write_bytes(b'local')
        return 0, ''
    monkeypatch.setattr(remote_render, 'try_remote_render', remote)
    monkeypatch.setattr(scene_blur, '_run_with_progress', local)
    out = tmp_path/'out.mkv'
    run(scene_blur._build_video_whole_file_reencode('ffmpeg', tmp_path/'original',
        [MuteInterval(2, 3)], 23, 10, 90, 8, 'hevc', out,
        source_probe={'streams': [{'codec_type': 'video', 'width': 1920, 'height': 1080, 'pix_fmt': 'yuv420p'}]}))
    assert calls == (['remote'] if remote_ok else ['remote', 'server CPU'])
    assert out.read_bytes() == (b'remote' if remote_ok else b'local')
