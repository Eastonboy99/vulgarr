"""Provider/checkpoint contract tests. Responses are mocked, not model qualification."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.vision import content_classifier as classifier
from app.scenes.content_policy import VISUAL_CATEGORIES


def observation(**updates):
    return {"schema_version": 1, "labels": {"clothing.bikini": .95},
            "coverage": sorted(VISUAL_CATEGORIES), "uncertain": False,
            "explanation": "Synthetic two-piece observation", "model": "mock-v1", **updates}


def test_whole_title_proposals_resume_contiguous_prefix_and_invalidate_source(tmp_path, monkeypatch):
    calls = []
    fail = True
    async def classify(**options):
        calls.append(options["start"])
        if options["start"] == 10 and fail:
            raise ValueError("simulated provider outage")
        return observation()
    monkeypatch.setattr(classifier, "classify_window", classify)
    options = dict(video_path=tmp_path/"title.mkv", duration=25, window_seconds=10,
                   checkpoint_dir=tmp_path/"work", fingerprint="source-v1", base_url="https://private",
                   api_key="secret", model="mock-v1", ffmpeg_bin="ffmpeg")
    with pytest.raises(ValueError):
        asyncio.run(classifier.scan_content_windows(**options))
    assert calls == [0, 10]
    fail = False
    calls.clear()
    results = asyncio.run(classifier.scan_content_windows(**options))
    assert calls == [10, 20]  # First successful window retained, no holes skipped.
    assert [(a, b) for a, b, _ in results] == [(0, 10), (10, 20), (20, 25)]
    calls.clear()
    asyncio.run(classifier.scan_content_windows(**{**options, "fingerprint": "source-v2"}))
    assert calls == [0, 10, 20]
    content = (tmp_path/"work"/"content-v1.jsonl").read_text()
    assert "secret" not in content and "https://private" not in content


def test_corrupt_checkpoint_tail_is_truncated_not_skipped(tmp_path, monkeypatch):
    calls = []
    async def classify(**options):
        calls.append(options["start"])
        return observation(labels={})
    monkeypatch.setattr(classifier, "classify_window", classify)
    options = dict(video_path=tmp_path/"t", duration=20, window_seconds=10,
                   checkpoint_dir=tmp_path, fingerprint="source", base_url="http://local",
                   api_key="", model="mock-v1", ffmpeg_bin="ffmpeg")
    asyncio.run(classifier.scan_content_windows(**options))
    checkpoint = tmp_path/"content-v1.jsonl"
    lines = checkpoint.read_text().splitlines()
    checkpoint.write_text(lines[0] + '\n{broken\n' + lines[1] + '\n')
    calls.clear()
    asyncio.run(classifier.scan_content_windows(**options))
    assert calls == [10]
    assert len(checkpoint.read_text().splitlines()) == 2


@pytest.mark.parametrize("reply,valid", [
    ({"labels": {"clothing.low_cut": .9}, "uncertain": False, "explanation": "Covered neckline"}, True),
    ({"labels": {"clothing.bikini": .9}, "uncertain": True, "explanation": "Ambiguous clothing"}, True),
    ({"labels": {"bathroom": .9}, "uncertain": False, "explanation": "A room"}, False),
    ({"labels": {}, "uncertain": "no", "explanation": "Wrong types"}, False),
    ({"labels": {}, "uncertain": False, "explanation": "Safe", "instructions": "extra"}, False),
])
def test_strict_provider_response_and_multiple_frames(tmp_path, monkeypatch, reply, valid):
    images = []
    requests = []
    def extract(binary, video, timestamp, target, **options):
        images.append(timestamp)
        target.write_bytes(b"synthetic frame, not real image")
    class Client:
        def __init__(self, **options):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **options):
            requests.append(options)
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": json.dumps(reply)}}]})
    monkeypatch.setattr(classifier, "extract_frame", extract)
    monkeypatch.setattr(classifier.httpx, "AsyncClient", Client)
    args = dict(base_url="http://provider/v1", api_key="secret", model="mock-v1", ffmpeg_bin="ffmpeg",
                video_path=tmp_path/"original", start=0, end=10)
    if valid:
        result = asyncio.run(classifier.classify_window(**args))
        assert result["labels"] == reply["labels"]
        assert set(result["coverage"]) == VISUAL_CATEGORIES
    else:
        with pytest.raises(ValueError, match="invalid evidence"):
            asyncio.run(classifier.classify_window(**args))
    assert images == [1, 5, 9]
    assert len(requests[0]["json"]["messages"][0]["content"]) == 4
