"""Truncation/error regressions. Never accept a partial model response as safe."""
import asyncio
import json
import socket

import httpx
import pytest

from app.vision import contextual as c


def complete_evidence():
    return c.observations_to_evidence({"frames": [{"frame_id": f"f{i}", "state": "contextual_nudity", "evidence": "Synthetic visible evidence"} for i in range(4)], "window_summary": "Synthetic summary"}, "mock")


@pytest.mark.parametrize("error,expected", [(httpx.ReadTimeout, "timed out"), (httpx.ConnectError, "could not connect"), (RuntimeError, "invalid evidence or configuration")])
def test_diagnostics_are_actionable_without_provider_payloads(tmp_path, monkeypatch, error, expected):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a: [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 1234))])
    def extract(binary, video, timestamp, target, **kwargs): target.write_bytes(b"synthetic")
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): raise error("secret-provider-payload https://private/?token=credential")
    monkeypatch.setattr(c, "extract_frame", extract)
    monkeypatch.setattr(c.httpx, "AsyncClient", Client)
    with pytest.raises(c.ContextualAnalysisError) as caught:
        asyncio.run(c.classify_window(base_url="http://localhost:1234/v1", api_key="private-key", model="mock", ffmpeg_bin="ffmpeg", video_path=tmp_path/"source", start=0, end=8))
    text = str(caught.value)
    assert expected in text and "Scan not activated" in text
    assert all(secret not in text for secret in ("credential", "secret-provider", "https://", "private-key"))


def test_failed_window_identified_and_never_checkpointed_or_skipped(tmp_path, monkeypatch):
    async def local(*args): return None
    calls, fail = [], [True]
    async def classify(**options):
        calls.append(options["start"])
        if options["start"] == 8 and fail[0]:
            raise c.ContextualAnalysisError("Local contextual analysis failed: model response was truncated; incomplete JSON was rejected")
        return complete_evidence()
    monkeypatch.setattr(c, "local_endpoint", local)
    monkeypatch.setattr(c, "classify_window", classify)
    args = dict(video_path=tmp_path/"source", duration=24, checkpoint_dir=tmp_path, fingerprint="source", base_url="http://localhost/v1", api_key="", model="mock", ffmpeg_bin="ffmpeg")
    with pytest.raises(c.ContextualAnalysisError, match=r"Window 2/3 \(8.00–16.00s\).*truncated"):
        asyncio.run(c.scan_contextual_windows(**args))
    saved = (tmp_path/(c.VERSION+".jsonl")).read_text().splitlines()
    assert len(saved) == 1 and json.loads(saved[0])["index"] == 0
    assert calls == [0, 8]
    fail[0] = False
    calls.clear()
    results = asyncio.run(c.scan_contextual_windows(**args))
    assert calls == [8, 16]
    assert len(results) == 3
