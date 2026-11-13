"""Settings/review contracts; no model calls or media writes."""
import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from app.db.session import DEFAULT_SETTINGS
from app.routers import settings, setup
from app.scenes.content_policy import DEFAULT_POLICY


def test_policy_settings_roundtrip_export_import_and_validation(monkeypatch):
    stored = {}
    @asynccontextmanager
    async def session():
        yield None
    async def get_setting(_session, key):
        return stored.get(key, DEFAULT_SETTINGS.get(key))
    async def set_setting(_session, key, value):
        stored[key] = value
    from app.scenes import protection
    monkeypatch.setattr(protection, "get_setting", get_setting)
    for module in (settings, setup):
        monkeypatch.setattr(module, "get_session", session)
        monkeypatch.setattr(module, "set_setting", set_setting)
        monkeypatch.setattr(module, "get_setting", get_setting)
    app = FastAPI()
    app.include_router(settings.router)
    app.include_router(setup.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            policy = copy.deepcopy(DEFAULT_POLICY)
            policy["actions"]["clothing.bikini"] = "blur"
            form = {"concurrency_cap": "1", "off_hours_start": "01:00", "off_hours_end": "07:00",
                    "scene_content_policy": json.dumps(policy)}
            response = await client.post("/settings", data=form)
            assert response.status_code == 200
            assert "Content preferences" in response.text
            assert "Local detector · 18 labels" in response.text
            from app.vision.nudenet_labels import NUDENET_LABELS
            assert response.text.count('data-policy-category="nudenet.') == 18
            for label in NUDENET_LABELS:
                assert f'data-policy-category="nudenet.{label}"' in response.text
                assert f'for="policy-{label}"' in response.text
            assert 'data-policy-category="nudenet.FEMALE_BREAST_COVERED"' in response.text
            assert 'data-policy-category="clothing.bikini"' not in response.text
            assert "Unavailable" not in response.text
            assert stored["scene_content_policy"] == policy
            exported = (await client.get("/settings/export")).json()
            assert exported["scene_content_policy"] == policy
            assert (await client.post("/setup/import", json=exported)).status_code == 200
            before = copy.deepcopy(stored)
            policy["actions"]["medical.graphic"] = "blur_mute"
            assert (await client.post("/settings", data={**form, "scene_content_policy": json.dumps(policy)})).status_code == 422
            assert (await client.post("/setup/import", json={"scene_content_policy": policy})).status_code == 400
            assert stored == before
            assert (await client.post("/settings", data={**form, "scene_semantic_enabled": "true"})).status_code == 422
            assert stored == before
    asyncio.run(check())


def test_review_template_explains_effective_actions_and_inheritance():
    from app.routers.scenes import templates
    from app.scenes.content_policy import CATEGORIES, scene_decision
    from app.domain import SceneReviewStatus
    scene = SimpleNamespace(id=123, start_seconds=1, end_seconds=2,
        source="detected", status=SceneReviewStatus.approved, peak_confidence=.9,
        verified_fraction=None, claude_verify_reason=None, audio_override=None,
        action_override=None, category_evidence=json.dumps({"schema_version": 1,
            "labels": {"clothing.bikini": .95},
            "coverage": sorted(k for k, (_, capability) in CATEGORIES.items() if capability == "visual"),
            "uncertain": False, "model": "mock-v1", "explanation": "Mock two-piece clothing"}),
        mute_audio=False, applied_at=None)
    scene.effective_decision = scene_decision(scene, DEFAULT_POLICY)
    scene.evidence_display = json.loads(scene.category_evidence)
    html = templates.get_template("partials/scene_review_list.html").render(
        title_id=1, scenes=[scene], category_labels=CATEGORIES,
        category_counts={"clothing.bikini": 1}, scan_generation="42", scene_scan_status="scanned",
        scan_job=None, blur_job=None, claude_verify_job=None, high_confidence_fraction=.5,
        pending_count=0, high_confidence_pending_count=0, detail_view=False,
        claude_vision_verify_enabled=False, policy_stale=True, scene_set_stale=True,
        scene_analysis=None, needs_reanalysis=False, legacy_evidence=False,
        vulgarr_edit_path="/edited.mkv")
    assert "Blur + mute" in html and "audio inherited" in html
    assert "mock-v1" in html and "Mock two-piece clothing" in html
    assert "Regenerate edit with current decisions" in html
    assert '"scan_generation": "42"' in html
    assert 'data-scene-categories="clothing.bikini"' in html


def test_missing_external_consent_rejected_before_queue(monkeypatch):
    from app.routers import scenes
    @asynccontextmanager
    async def session():
        yield None
    async def disabled(_session, key):
        return False
    monkeypatch.setattr(scenes, "get_session", session)
    monkeypatch.setattr(scenes, "get_setting", disabled)
    app = FastAPI()
    app.include_router(scenes.router)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            assert (await client.post("/library/title/1/reverify-scenes-claude")).status_code == 409
    asyncio.run(check())
