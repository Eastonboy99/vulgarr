"""First-run setup wizard -- reuses the existing Settings page's fields/save route
(app/routers/settings.py) rather than duplicating every integration/toggle field.
See app/auth.py's SetupWizardGateMiddleware for how a fresh install ends up here.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.db.session import get_session, get_setting, set_setting
from app.vision import contextual
from app.routers.settings import SETTING_KEYS, _load_all
from app.scenes.content_policy import validate_policy

router = APIRouter(prefix="/setup", tags=["setup"])
templates = Jinja2Templates(directory="app/templates")


def _filter_importable_settings(data) -> dict:
    """Whitelist-filters an uploaded config dict down to recognized setting keys --
    reuses SETTING_KEYS, the same whitelist app.routers.settings.update_settings
    itself trusts, so an imported file can't inject an arbitrary/unknown key into
    app_settings. No per-field type validation beyond this -- the only realistic
    input is a file this same app exported (see settings.py's GET /settings/export)."""
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in SETTING_KEYS}


@router.get("", response_class=HTMLResponse)
async def setup_wizard(request: Request):
    values = await _load_all()
    return templates.TemplateResponse(
        "setup.html", {"request": request, "values": values, "wizard_mode": True}
    )


@router.post("/complete")
async def complete_setup():
    async with get_session() as session:
        await set_setting(session, "setup_wizard_completed", True)
    return RedirectResponse(url="/library", status_code=303)


@router.post("/import")
async def import_config(request: Request):
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 -- malformed upload, not a server error
        return JSONResponse({"ok": False, "error": "Not valid JSON"}, status_code=400)

    filtered = _filter_importable_settings(data)
    if not filtered:
        return JSONResponse(
            {"ok": False, "error": "No recognized settings found in that file"}, status_code=400
        )

    try:
        if "scene_content_policy" in filtered:
            filtered["scene_content_policy"] = validate_policy(filtered["scene_content_policy"])
        if "scene_semantic_enabled" in filtered and type(filtered["scene_semantic_enabled"]) is not bool:
            raise ValueError("Semantic opt-in must be Boolean")
        if "scene_semantic_window_seconds" in filtered:
            window = filtered["scene_semantic_window_seconds"]
            if type(window) not in (int, float) or not 1 <= window <= 30:
                raise ValueError("Semantic window must be between 1 and 30 seconds")
        from app.scenes import protection
        if "scene_protection_enabled" in filtered and type(filtered["scene_protection_enabled"]) is not bool:
            raise ValueError("Protection-first opt-in must be Boolean")
        if set(filtered) & (set(contextual.KEYS) | set(protection.KEYS) | {"scene_semantic_enabled", "movie_output_folder", "tv_output_folder"}):
            async with get_session() as session:
                config = {key: filtered.get(key, await get_setting(session, key)) for key in contextual.KEYS}
                protected = filtered.get("scene_protection_enabled", await protection.enabled(session))
                if protected:
                    semantic = filtered.get("scene_semantic_enabled", await get_setting(session, "scene_semantic_enabled"))
                    folders = [filtered.get(k, await get_setting(session, k)) for k in ("movie_output_folder", "tv_output_folder")]
                    if semantic or not any(isinstance(f, str) and f.strip() for f in folders):
                        raise ValueError("Protection-first mode requires clean output folders and no legacy semantic analysis")
                    config["scene_contextual_enabled"] = True
            await contextual.validate_config(config)
    except (ValueError, TypeError):
        return JSONResponse({"ok": False, "error": "Invalid or unsupported content policy settings"}, status_code=400)
    async with get_session() as session:
        for key, value in filtered.items():
            await set_setting(session, key, value)
    return JSONResponse({"ok": True, "imported": len(filtered)})
