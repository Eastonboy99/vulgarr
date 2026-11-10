"""Transport/scan regressions without loading NudeNet, CUDA or real media."""
import asyncio
import functools


def run_async(function):
    """Match the suite's asyncio.run style without requiring pytest-asyncio."""
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run
import importlib.util
import json
import sys
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.routers.remote_workers import byte_range, router
from app.vision import classifier
from app.vision.remote_source import remote_extract_enabled, source_signature


@pytest.mark.parametrize('header,size,want', [
    (None, 100, (0, 99)), ('bytes=10-19', 100, (10, 19)),
    ('bytes=90-', 100, (90, 99)), ('bytes=-10', 100, (90, 99)),
    ('bytes=0-1000', 100, (0, 99)), ('bytes=-200', 100, (0, 99)),
])
def test_ranges(header, size, want):
    assert byte_range(header, size) == want


@pytest.mark.parametrize('header', ['bytes=100-', 'bytes=20-10', 'bytes=-0',
                                   'bytes=0-1,4-5', 'garbage', 'bytes=-'])
def test_bad_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 100)


@run_async
async def test_signed_source_ranges(monkeypatch, tmp_path):
    import time
    monkeypatch.setenv('SPF_INTERNAL_TOKEN', 'test-secret')
    file = tmp_path / 'movie.mkv'
    file.write_bytes(b'0123456789')
    app = FastAPI()
    app.include_router(router)
    expires = int(time.time()) + 60
    params = {'path': str(file), 'expires': expires,
              'signature': source_signature(str(file), expires)}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        response = await client.get('/internal/source', params=params, headers={'Range': 'bytes=2-5'})
        assert response.status_code == 206
        assert response.content == b'2345'
        assert response.headers['Content-Range'] == 'bytes 2-5/10'
        response = await client.head('/internal/source', params=params)
        assert response.content == b''
        assert response.headers['content-length'] == '10'
        params['path'] = str(tmp_path / 'another-file')
        assert (await client.get('/internal/source', params=params)).status_code == 403
        params['path'] = str(file)
        params['expires'] = 0
        assert (await client.get('/internal/source', params=params)).status_code == 403


def configure_remote(monkeypatch):
    monkeypatch.setenv('SPF_REMOTE_EXTRACT', '1')
    monkeypatch.setenv('SPF_CLASSIFIER_URLS', 'http://worker')
    monkeypatch.setenv('SPF_INTERNAL_BASE_URL', 'http://vulgarr')
    monkeypatch.setenv('SPF_INTERNAL_TOKEN', 'secret')


@pytest.mark.parametrize('detections', [None, {}, [{}], [{'class': 'FACE_FEMALE'}],
    [{'class': 'FACE_FEMALE', 'score': float('nan')}],
    [{'class': 'FEMALE_BREAST_EXPOSED', 'score': 2}]])
def test_invalid_remote_detections_are_not_zero_scores(detections):
    with pytest.raises(ValueError):
        classifier._validate_remote_detections(detections)


def test_remote_requires_complete_configuration(monkeypatch):
    configure_remote(monkeypatch)
    assert remote_extract_enabled()
    monkeypatch.delenv('SPF_INTERNAL_TOKEN')
    assert not remote_extract_enabled()


@run_async
async def test_double_buffer_and_checkpoint(monkeypatch, tmp_path):
    monkeypatch.delenv('SPF_CLASSIFIER_URLS', raising=False)
    monkeypatch.delenv('SPF_REMOTE_EXTRACT', raising=False)
    monkeypatch.setattr(classifier, '_SCAN_CHUNK_SECONDS', 2)
    extracting_second = asyncio.Event()
    states = []
    async def extract(*args, **kwargs):
        if kwargs['start_offset'] == 2:
            extracting_second.set()
        for i in range(2):
            (args[3] / f'frame_{i+1:08d}.jpg').write_bytes(b'jpeg')
        await kwargs['on_extract_progress'](1)
    async def detect(path, local):
        await asyncio.wait_for(extracting_second.wait(), 2)
        return []
    async def status(message):
        states.append(message)
    monkeypatch.setattr(classifier, '_extract_frames_continuous', extract)
    monkeypatch.setattr(classifier, '_detect', detect)
    checkpoint = tmp_path / 'scores.jsonl'
    scores = await classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 4, 1,
                                                resume_scores_path=checkpoint, on_status=status)
    assert sorted(s.timestamp for s in scores) == [0, 1, 2, 3]
    assert sorted(json.loads(l)['t'] for l in checkpoint.read_text().splitlines()) == [0, 1, 2, 3]
    assert any(s.startswith('Extracting chunk') for s in states)
    assert any(s.startswith('Classifying') for s in states)


@run_async
async def test_remote_chunk_skips_local_and_validates_all_scores(monkeypatch, tmp_path):
    configure_remote(monkeypatch)
    async def handle(request):
        return httpx.Response(200, json={'scores': [{'t': 0, 'c': .9}, {'t': 1, 'c': .1}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(classifier, '_remote_http_client', lambda: client)
        async def forbidden(*args, **kwargs):
            pytest.fail('local extraction should be skipped')
        monkeypatch.setattr(classifier, '_extract_frames_continuous', forbidden)
        scores = await classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 2, 1,
                                                    resume_scores_path=tmp_path/'scores')
    assert [s.confidence for s in scores] == [.9, .1]


@run_async
async def test_remote_incomplete_chunk_falls_back_under_lock(monkeypatch, tmp_path):
    from app.queue.heavy_pipeline_lock import HEAVY_PIPELINE_LOCK
    configure_remote(monkeypatch)
    async def handle(request):
        return httpx.Response(200, json={'scores': [{'t': 0, 'c': .9}]})
    async def extract(*args, **kwargs):
        assert HEAVY_PIPELINE_LOCK.locked()
        for i in range(2):
            (args[3]/f'frame_{i+1:08d}.jpg').write_bytes(b'jpeg')
    async def detect(*args):
        return []
    monkeypatch.setattr(classifier, '_extract_frames_continuous', extract)
    monkeypatch.setattr(classifier, '_detect', detect)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(classifier, '_remote_http_client', lambda: client)
        scores = await classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 2, 1,
                                                    resume_scores_path=tmp_path/'scores')
    assert [s.timestamp for s in scores] == [0, 1]
    assert not HEAVY_PIPELINE_LOCK.locked()


@run_async
async def test_worker_batch_order_and_stats_without_gpu(monkeypatch):
    fake = types.ModuleType('nudenet')
    fake.NudeDetector = object
    monkeypatch.setitem(sys.modules, 'nudenet', fake)
    spec = importlib.util.spec_from_file_location('worker_test', Path('remote_worker/server.py'))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    class Detector:
        def detect(self, path):
            return [{'class': 'test', 'score': float(Path(path).read_bytes())}]
    monkeypatch.setattr(worker, 'detector', lambda: Detector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(worker.app), base_url='http://worker') as client:
        response = await client.post('/detect-batch', files=[('frames', ('a.jpg', b'0.1')),
                                                           ('frames', ('b.jpg', b'0.9'))])
        assert [r[0]['score'] for r in response.json()['detections']] == [.1, .9]
        stats = (await client.get('/stats')).json()
        assert stats['frames_served'] == 2
        assert stats['requests_served'] == 1
        assert stats['in_flight'] == 0
        assert (await client.post('/detect-batch', files=[('frames', ('a', b'1'))]*17)).status_code == 422
        assert (await client.post('/encode', files={'segment': ('s.mkv', b'data')},
                                  data={'params': '{}'})).status_code == 422
        def no_gpu(*args):
            raise RuntimeError('No NVENC device')
        real_encode = worker._encode_segment
        monkeypatch.setattr(worker, '_encode_segment', no_gpu)
        params = {'source_codec': 'hevc', 'crf': 23, 'preset': 'fast',
                  'filter_complex': '[0:v]null[vb]'}
        assert (await client.post('/encode', files={'segment': ('s.mkv', b'data')},
                                  data={'params': json.dumps(params)})).status_code == 503
        monkeypatch.setattr(worker, '_encode_segment', real_encode)
        commands = []
        def encode_run(command, **kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b'encoded-mkv')
            return types.SimpleNamespace(returncode=0)
        monkeypatch.setattr(worker.subprocess, 'run', encode_run)
        from app.mux.scene_blur import build_blur_filter
        from app.audio.mute import MuteInterval
        params['filter_complex'] = build_blur_filter([MuteInterval(1, 2)], '0:v', 'vb')
        response = await client.post('/encode', files={'segment': ('s.mkv', b'data')},
                                     data={'params': json.dumps(params)})
        assert response.status_code == 200
        assert response.content == b'encoded-mkv'
        assert 'hevc_nvenc' in commands[0]
        assert params['filter_complex'] in commands[0]
        params['filter_complex'] = 'movie=/etc/passwd[vb]'
        assert (await client.post('/encode', files={'segment': ('s.mkv', b'data')},
                                  data={'params': json.dumps(params)})).status_code == 503
        assert len(commands) == 1


@run_async
async def test_malformed_single_response_retries_local(monkeypatch, tmp_path):
    monkeypatch.setenv('SPF_CLASSIFIER_URLS', 'http://worker')
    frame = tmp_path / 'frame.jpg'
    frame.write_bytes(b'jpeg')
    async def handle(request):
        return httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(classifier, '_remote_http_client', lambda: client)
        result = await classifier._detect(frame, lambda path: [{'class': 'local', 'score': 1}])
    assert result[0]['class'] == 'local'


@run_async
async def test_encode_failure_preserves_local_codec_path(monkeypatch, tmp_path):
    from app.mux import scene_blur
    from app.audio.mute import MuteInterval
    monkeypatch.setenv('SPF_REMOTE_ENCODE_URL', 'http://worker')
    calls = []
    async def run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b'clip')
        return 0, '', ''
    async def atomic(build, output, timeout, message):
        calls.append(build(output))
        output.write_bytes(b'local-result')
    async def source_params(*args):
        return None
    async def handle(request):
        return httpx.Response(503)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(scene_blur.httpx, 'AsyncClient', lambda **kwargs: client)
    monkeypatch.setattr(scene_blur, '_run', run)
    monkeypatch.setattr(scene_blur, '_run_atomic', atomic)
    monkeypatch.setattr(scene_blur, '_probe_source_hevc_params', source_params)
    plan = scene_blur.VideoSegmentPlan(0, 10, True, (MuteInterval(1, 2),))
    output = tmp_path / 'out.mkv'
    backend = await scene_blur._extract_reencode_segment('ffmpeg', tmp_path/'video', plan,
                                                        23, 'fast', 90, 8, output, 'hevc')
    assert backend == 'CPU'
    assert calls[-1][calls[-1].index('-c:v') + 1] == 'libx265'
    assert output.read_bytes() == b'local-result'


@run_async
async def test_nvenc_hevc_parameter_mismatch_is_rejected(monkeypatch, tmp_path):
    from app.mux import scene_blur
    monkeypatch.setenv('SPF_REMOTE_ENCODE_URL', 'http://worker')
    async def run(command, **kwargs):
        if '-show_entries' in command:
            return 0, json.dumps({'streams': [{'codec_name': 'hevc', 'width': 1920,
                'height': 1080, 'pix_fmt': 'yuv420p', 'start_time': '0'}],
                'format': {'duration': '10'}}), ''
        Path(command[-1]).write_bytes(b'clip')
        return 0, '', ''
    async def probe(ffmpeg, path, at):
        return 'different-nvenc-state' if path.name == 'encoded.mkv' else 'source-state'
    async def handle(request):
        return httpx.Response(200, content=b'encoded')
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(scene_blur.httpx, 'AsyncClient', lambda **kwargs: client)
    monkeypatch.setattr(scene_blur, '_run', run)
    monkeypatch.setattr(scene_blur, '_probe_source_hevc_params', probe)
    output = tmp_path / 'result.mkv'
    assert not await scene_blur._try_remote_reencode('ffmpeg', tmp_path/'video',
        scene_blur.VideoSegmentPlan(0, 10, True), 23, 'fast', 90, 8, output, 'hevc')
    assert not output.exists()


@run_async
async def test_scan_cancellation_joins_extract_and_cleans_tempdirs(monkeypatch, tmp_path):
    monkeypatch.delenv('SPF_REMOTE_EXTRACT', raising=False)
    monkeypatch.delenv('SPF_CLASSIFIER_URLS', raising=False)
    running = asyncio.Event()
    stopped = asyncio.Event()
    paths = []
    async def extract(*args, **kwargs):
        paths.append(args[3])
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    monkeypatch.setattr(classifier, '_extract_frames_continuous', extract)
    task = asyncio.create_task(classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 2, 1))
    await running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert all(not p.exists() for p in paths)


@run_async
async def test_client_batches_preserve_frames_and_bound_requests(monkeypatch, tmp_path):
    monkeypatch.delenv('SPF_REMOTE_EXTRACT', raising=False)
    monkeypatch.setenv('SPF_CLASSIFIER_URLS', 'http://worker')
    sizes = []
    active = peak = 0
    async def extract(*args, **kwargs):
        for i in range(25):
            (args[3] / f'frame_{i+1:08d}.jpg').write_bytes(b'jpeg')
    async def handle(request):
        nonlocal active, peak
        assert request.url.path == '/detect-batch'
        count = request.content.count(b'name="frames"')
        sizes.append(count)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001)
        active -= 1
        return httpx.Response(200, json={'detections': [[] for _ in range(count)]})
    monkeypatch.setattr(classifier, '_extract_frames_continuous', extract)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(classifier, '_remote_http_client', lambda: client)
        scores = await classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 25, 1,
                                                    concurrency=2, resume_scores_path=tmp_path/'scores')
    assert sorted(sizes) == [1, 12, 12]
    assert peak == 2
    assert sorted(s.timestamp for s in scores) == list(range(25))


@run_async
async def test_remote_chunks_overlap_but_checkpoint_in_source_order(monkeypatch, tmp_path):
    configure_remote(monkeypatch)
    monkeypatch.setattr(classifier, '_SCAN_CHUNK_SECONDS', 2)
    second_started = asyncio.Event()
    async def handle(request):
        start = json.loads(request.content)['start']
        if start == 2:
            second_started.set()
        else:
            await asyncio.wait_for(second_started.wait(), 2)
        return httpx.Response(200, json={'scores': [{'t': start, 'c': .1}, {'t': start+1, 'c': .2}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(classifier, '_remote_http_client', lambda: client)
        checkpoint = tmp_path / 'scores'
        await classifier.scan_video_frames(tmp_path/'video', 'ffmpeg', 4, 1,
                                            resume_scores_path=checkpoint)
    assert [json.loads(l)['t'] for l in checkpoint.read_text().splitlines()] == [0, 1, 2, 3]
