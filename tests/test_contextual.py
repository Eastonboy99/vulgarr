"""Local discovery contracts; synthetic observations are NOT accuracy claims."""
import asyncio
import copy
import json
import socket
from types import SimpleNamespace

import pytest

from app.scenes.content_policy import (DEFAULT_POLICY, CONTEXTUAL_CATEGORIES,
    resolve_action, validate_policy)
from app.vision import contextual as c


def observation(state="contextual_nudity"):
    return {"frames": [{"frame_id": f"f{i}", "state": state, "evidence": "Synthetic visible evidence"} for i in range(4)], "window_summary": "Synthetic summary"}


def evidence(state="contextual_nudity", model="mock"):
    return c.observations_to_evidence(observation(state), model)


@pytest.fixture
def private_dns(monkeypatch):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.23", 1234))])


def test_schema2_policy_upgrade_preserves_actions_and_adds_contextual():
    old = copy.deepcopy(DEFAULT_POLICY)
    old["schema_version"] = 2
    del old["actions"]["nudity.contextual"]
    old["actions"]["nudenet.FEMALE_BREAST_EXPOSED"] = "allow"
    new = validate_policy(old)
    assert new["schema_version"] == 3
    assert all(new["actions"][k] == v for k, v in old["actions"].items())
    assert new["actions"]["nudity.contextual"] == "blur_mute"


@pytest.mark.parametrize("state,positive,uncertain", [
    ("contextual_nudity", True, False), ("exposed_nudity", True, False),
    ("towel_sheet_covered", False, False), ("swimwear_underwear", False, False),
    ("clothed", False, False), ("no_people", False, False),
    ("uncertain", False, True), ("ambiguous_bare_skin", False, True)])
def test_evidence_is_binary_uncertainty_derived_and_review_required(state, positive, uncertain):
    e = evidence(state)
    assert bool(e["labels"]) is positive
    assert e["uncertain"] is uncertain
    assert set(e["coverage"]) == CONTEXTUAL_CATEGORIES
    assert resolve_action(DEFAULT_POLICY, e).needs_review
    assert not resolve_action(DEFAULT_POLICY, e, action_override="blur").needs_review


@pytest.mark.parametrize("failure", ["duplicate", "missing", "timestamp", "state", "review", "extra", "summary", "long_summary", "text"])
def test_rejects_untrusted_frame_ids_timestamps_and_decisions(failure):
    raw = observation()
    if failure == "duplicate": raw["frames"][1]["frame_id"] = "f0"
    if failure == "missing": raw["frames"].pop()
    if failure == "timestamp": raw["frames"][0]["timestamp"] = 2206
    if failure == "state": raw["frames"][0]["state"] = "safe"
    if failure == "review": raw["needs_review"] = False
    if failure == "extra": raw["frames"][0]["approve"] = True
    if failure == "summary": raw["window_summary"] = ""
    if failure == "long_summary": raw["window_summary"] = "x" * (c.MAX_SUMMARY_CHARS + 1)
    if failure == "text": raw["frames"][0]["evidence"] = "x" * (c.MAX_EVIDENCE_CHARS + 1)
    with pytest.raises(ValueError): c.observations_to_evidence(raw, "mock")


@pytest.mark.parametrize("url", ["https://public/v1", "ftp://local/v1", "http://user:secret@local/v1", "http://local/v1?key=secret", "http://local/v1#fragment", ""])
def test_rejects_nonlocal_or_credential_urls(monkeypatch, url):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", ("8.8.8.8", 1234))])
    with pytest.raises(ValueError, match="local/private"):
        asyncio.run(c.local_endpoint(url))


def test_dns_pinning_no_second_hostname_lookup(private_dns, monkeypatch):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a: [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 1234, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.23", 1234))])
    target, headers, extensions = asyncio.run(c.local_endpoint("http://desktop:1234/v1/"))
    assert str(target) == "http://192.168.1.23:1234/v1/chat/completions"
    assert headers == {"Host": "desktop:1234"}
    assert extensions["sni_hostname"] == "desktop"


def test_mixed_private_public_dns_rejected(monkeypatch):
    monkeypatch.setattr(c.socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", (ip, 1234)) for ip in ("192.168.1.23", "8.8.8.8")])
    with pytest.raises(ValueError): asyncio.run(c.local_endpoint("http://desktop/v1"))


@pytest.mark.parametrize("status,finish,valid", [(200, "stop", True), (302, "stop", False), (200, "length", False), (500, "stop", False),
    (200, "content_filter", False), (200, "bad_json", False), (200, "bad_observations", False)])
def test_request_is_local_strict_and_failures_do_not_leak(tmp_path, monkeypatch, private_dns, status, finish, valid):
    calls, times, constructors = [], [], []
    def extract(binary, video, timestamp, target, **options):
        times.append(timestamp)
        assert options["max_long_edge"] == 768
        target.write_bytes(b"synthetic image")
    class Client:
        def __init__(self, **options): constructors.append(options)
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **options):
            calls.append((url, options))
            raw = observation()
            if finish == "bad_observations": raw["frames"][0]["frame_id"] = "unknown"
            text = '{"frames":' if finish == "bad_json" else json.dumps(raw)
            completed = "stop" if finish in ("bad_json", "bad_observations") else finish
            return SimpleNamespace(status_code=status, json=lambda: {"choices": [{"finish_reason": completed, "message": {"content": text}}]})
    monkeypatch.setattr(c, "extract_frame", extract)
    monkeypatch.setattr(c.httpx, "AsyncClient", Client)
    args = dict(base_url="http://desktop:1234/v1", api_key="", model="mock", ffmpeg_bin="ffmpeg", video_path=tmp_path/"source", start=0, end=8)
    if valid:
        assert asyncio.run(c.classify_window(**args))["labels"] == {"nudity.contextual": 1.0}
    else:
        with pytest.raises(ValueError) as err: asyncio.run(c.classify_window(**args))
        assert "desktop" not in str(err.value)
        if finish == "length": assert "truncated" in str(err.value)
        if finish == "bad_json": assert "malformed or incomplete JSON" in str(err.value)
        if finish == "bad_observations": assert "invalid structured observations" in str(err.value)
        if status != 200: assert f"HTTP {status}" in str(err.value)
    assert times == [1, 3, 5, 7]
    assert constructors[0]["trust_env"] is False
    assert constructors[0]["follow_redirects"] is False
    assert "Authorization" not in calls[0][1]["headers"]
    payload = calls[0][1]["json"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == c.MAX_OUTPUT_TOKENS == 2048
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["frames"]["items"]["properties"]["evidence"]["maxLength"] == c.MAX_EVIDENCE_CHARS == 240
    assert schema["properties"]["window_summary"]["maxLength"] == c.MAX_SUMMARY_CHARS == 240
    assert len(payload["messages"][0]["content"]) == 9


def test_resume_only_contiguous_prefix_and_configuration_identity(tmp_path, monkeypatch, private_dns):
    calls, fail = [], [True]
    async def classify(**options):
        calls.append(options["start"])
        if options["start"] == 8 and fail[0]: raise ValueError("outage")
        return evidence(model=options["model"])
    monkeypatch.setattr(c, "classify_window", classify)
    args = dict(video_path=tmp_path/"source", duration=20, checkpoint_dir=tmp_path,
        fingerprint="source-v1", base_url="http://desktop:1234/v1", api_key="private-secret", model="mock", ffmpeg_bin="ffmpeg")
    with pytest.raises(ValueError): asyncio.run(c.scan_contextual_windows(**args))
    assert calls == [0, 8]
    fail[0] = False
    calls.clear()
    results = asyncio.run(c.scan_contextual_windows(**args))
    assert calls == [8, 16]
    assert [(a, b) for a, b, _ in results] == [(0, 8), (8, 16), (16, 20)]
    saved = (tmp_path/(c.VERSION+".jsonl")).read_text()
    assert "private-secret" not in saved and "http://desktop" not in saved
    for change in ({"fingerprint": "source-v2"}, {"model": "other"}, {"window_seconds": 10}):
        calls.clear()
        asyncio.run(c.scan_contextual_windows(**{**args, **change}))
        assert calls[0] == 0
    asyncio.run(c.scan_contextual_windows(**args))
    monkeypatch.setattr(c, "PROMPT", c.PROMPT + "\nRevised observation contract.")
    calls.clear()
    asyncio.run(c.scan_contextual_windows(**args))
    assert calls == [0, 8, 16]


def test_corrupt_checkpoint_tail_is_not_skipped(tmp_path, monkeypatch, private_dns):
    calls = []
    async def classify(**options):
        calls.append(options["start"])
        return evidence("clothed")
    monkeypatch.setattr(c, "classify_window", classify)
    args = dict(video_path=tmp_path/"source", duration=24, checkpoint_dir=tmp_path,
        fingerprint="source", base_url="http://desktop/v1", api_key="", model="mock", ffmpeg_bin="ffmpeg")
    asyncio.run(c.scan_contextual_windows(**args))
    path = tmp_path/(c.VERSION+".jsonl")
    lines = path.read_text().splitlines()
    path.write_text(lines[0]+"\n{broken\n"+lines[2]+"\n")
    calls.clear()
    asyncio.run(c.scan_contextual_windows(**args))
    assert calls == [8, 16]
    assert len(path.read_text().splitlines()) == 3


@pytest.mark.parametrize("window,timeout", [(0, 300), (float("nan"), 300), (8, float("inf")), (8, 1), (True, 300)])
def test_invalid_sampling_rejected(window, timeout):
    with pytest.raises(ValueError): c.validate_sampling(window, timeout)
