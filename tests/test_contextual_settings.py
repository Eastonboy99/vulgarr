"""UI/settings opt-in and local-only validation without external requests."""
import asyncio
import copy
from contextlib import asynccontextmanager
import json
import socket
from types import SimpleNamespace

from fastapi import FastAPI
import httpx

from app.db.session import DEFAULT_SETTINGS
from app.domain import SceneReviewStatus
from app.routers import settings, setup
from app.scenes.content_policy import CATEGORIES, DEFAULT_POLICY, scene_decision
from app.vision import contextual


def test_local_settings_roundtrip_optional_key_old_client_and_import(monkeypatch):
    stored = {}
    @asynccontextmanager
    async def session(): yield None
    async def get_setting(session, key): return stored.get(key, DEFAULT_SETTINGS[key])
    async def set_setting(session, key, value): stored[key] = value
    for module in (settings, setup):
        monkeypatch.setattr(module, "get_session", session)
        monkeypatch.setattr(module, "get_setting", get_setting)
        monkeypatch.setattr(module, "set_setting", set_setting)
    monkeypatch.setattr(contextual.socket, "getaddrinfo", lambda host, *args: [(socket.AF_INET, 1, 6, "", ("8.8.8.8" if host == "public" else "192.168.1.23", 1234))])
    app = FastAPI()
    app.include_router(settings.router)
    app.include_router(setup.router)
    base = {"concurrency_cap": "1", "off_hours_start": "01:00", "off_hours_end": "07:00"}
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            form = {**base, "scene_contextual_present": "true", "scene_contextual_enabled": "true", "scene_contextual_base_url": "http://desktop:1234/v1", "scene_contextual_model": "qwen3.5-4b"}
            response = await client.post("/settings", data=form)
            assert response.status_code == 200
            assert stored["scene_contextual_enabled"] is True
            assert stored["scene_contextual_api_key"] == ""
            assert response.text.count('data-policy-category="nudenet.') == 18
            assert "Discover contextual / obscured nudity across the whole video" in response.text
            assert "Verify existing legacy candidates" in response.text
            assert "Contextual nudity discovery" in response.text
            assert "costs real external money" not in response.text
            before = copy.deepcopy(stored)
            assert (await client.post("/settings", data={**form, "scene_contextual_base_url": "http://public/v1"})).status_code == 422
            assert stored == before
            for update in ({"scene_contextual_window_seconds": "nan"}, {"scene_contextual_timeout_seconds": "1"}):
                assert (await client.post("/settings", data={**form, **update})).status_code == 422
                assert stored == before
            # Old clients omit new controls: do not silently turn discovery off.
            assert (await client.post("/settings", data=base)).status_code == 200
            assert stored["scene_contextual_enabled"] is True
            exported = (await client.get("/settings/export")).json()
            assert (await client.post("/setup/import", json=exported)).status_code == 200
            before = copy.deepcopy(stored)
            for payload in ({"scene_contextual_enabled": "false"}, {"scene_contextual_base_url": "http://public/v1"}, {"scene_contextual_window_seconds": 0}):
                assert (await client.post("/setup/import", json=payload)).status_code == 400
                assert stored == before
            assert (await client.post("/settings", data={**base, "scene_contextual_present": "true"})).status_code == 200
            assert stored["scene_contextual_enabled"] is False
            # A local candidate verifier must not require a dummy cloud key.
            assert (await client.post("/settings", data={**base, "claude_vision_verify_enabled": "true", "claude_vision_base_url": "http://desktop:1234/v1", "claude_vision_model": "qwen3.5-4b"})).status_code == 200
            assert stored["claude_vision_verify_enabled"] is True
    asyncio.run(check())


def test_contextual_review_row_does_not_present_binary_flag_as_confidence():
    from app.routers.scenes import templates
    raw = contextual.observations_to_evidence({"frames": [{"frame_id": f"f{i}", "state": "contextual_nudity", "evidence": "Synthetic evidence"} for i in range(4)], "window_summary": "Synthetic summary"}, "mock")
    scene = SimpleNamespace(id=123, source="detected", status=SceneReviewStatus.pending,
        start_seconds=1, end_seconds=8, peak_confidence=1, verified_fraction=None,
        action_override=None, audio_override=None, mute_audio=False, claude_verify_reason=None,
        category_evidence=json.dumps(raw), evidence_display=raw)
    scene.effective_decision = scene_decision(scene, DEFAULT_POLICY)
    rendered = templates.get_template("partials/scene_review_row.html").render(scene=scene, title_id=1, category_labels=CATEGORIES, claude_vision_verify_enabled=True)
    assert "AI flag" in rendered and "review required" in rendered
    assert "100%" not in rendered
    assert "Not verified" not in rendered
