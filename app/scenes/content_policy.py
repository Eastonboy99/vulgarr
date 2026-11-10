"""Versioned, provider-independent content decisions.

Room names are deliberately absent: context alone never proves objectionable
content. Evidence is multi-label and policy independent; allowing one label
cannot cancel a different blocking label. This module performs no I/O.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum

SCHEMA_VERSION = 1
RENDER_SETTING_KEYS = (
    "scene_blur_radius", "scene_blur_power", "scene_blur_pad_start_seconds",
    "scene_blur_pad_end_seconds", "blur_video_crf", "blur_video_preset",
)


class Action(str, Enum):
    allow = "allow"
    blur = "blur"
    blur_mute = "blur_mute"


# Stable ID -> (display label, capability). Only the first two capabilities
# have an implemented semantic frame-analysis path. Temporal/transcript groups
# remain unavailable rather than presenting working-looking controls.
CATEGORIES = {
    "nudity.breasts": ("Exposed breasts", "visual"),
    "nudity.genitalia": ("Exposed genitalia", "visual"),
    "nudity.buttocks": ("Exposed buttocks", "visual"),
    "clothing.low_cut": ("Low-cut neckline / cleavage", "visual"),
    "clothing.chest_outline": ("Covered chest outline", "visual"),
    "clothing.bikini": ("Bikini / two-piece swimwear", "visual"),
    "clothing.one_piece": ("One-piece swimwear", "visual"),
    "clothing.underwear": ("Underwear / lingerie", "visual"),
    "clothing.male_tight_underwear": ("Male tight underwear", "visual"),
    "sexual.activity": ("Sexual activity", "temporal"),
    "sexual.implied": ("Implied sexual activity", "temporal"),
    "sexual.gestures": ("Sexual gestures", "temporal"),
    "violence.graphic": ("Graphic injury / gore", "unavailable"),
    "violence.physical": ("Non-graphic violence", "unavailable"),
    "violence.threats": ("Weapons / threats", "unavailable"),
    "disturbing.imagery": ("Disturbing imagery", "unavailable"),
    "self_harm.visual": ("Visual self-harm", "unavailable"),
    "self_harm.references": ("Spoken / written self-harm references", "transcript"),
    "medical.graphic": ("Graphic medical procedures", "unavailable"),
    "medical.non_graphic": ("Non-graphic medical / body processes", "unavailable"),
}
VISUAL_CATEGORIES = frozenset(k for k, (_, capability) in CATEGORIES.items() if capability == "visual")
DEFAULT_POLICY = {
    "schema_version": SCHEMA_VERSION,
    "mute_blurred_audio": True,
    "actions": {k: ("blur_mute" if k.startswith("nudity.") or k in {
        "clothing.bikini", "clothing.underwear", "clothing.male_tight_underwear"
    } else "allow") for k in CATEGORIES},
}


def validate_policy(value: dict) -> dict:
    """Return a canonical policy, rejecting unknown versions/IDs/actions.

    Unimplemented categories cannot be enabled, even by imported settings.
    Capability availability (provider opt-in) is checked by the caller.
    """
    if not isinstance(value, dict) or set(value) != {"schema_version", "mute_blurred_audio", "actions"}:
        raise ValueError("Invalid content policy fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported content policy schema")
    if type(value["mute_blurred_audio"]) is not bool:
        raise ValueError("Audio default must be Boolean")
    actions = value["actions"]
    if not isinstance(actions, dict) or set(actions) != set(CATEGORIES):
        raise ValueError("Policy must contain exactly the known category IDs")
    for category, action in actions.items():
        Action(action)
        if category not in VISUAL_CATEGORIES and action != "allow":
            raise ValueError(f"Category {category} has no qualified analysis backend")
    return {"schema_version": SCHEMA_VERSION, "mute_blurred_audio": value["mute_blurred_audio"],
            "actions": dict(sorted(actions.items()))}


def policy_fingerprint(policy: dict) -> str:
    """Stable render revision, independent of dictionary order."""
    return hashlib.sha256(json.dumps(validate_policy(policy), sort_keys=True).encode()).hexdigest()


def validate_evidence(value: dict) -> dict:
    """Validate a complete semantic response, including explicit uncertainty.

    Coverage is a statement about the categories examined, not a guarantee of
    recall between sampled frames. Missing labels within coverage mean absent.
    """
    if not isinstance(value, dict) or set(value) != {"schema_version", "labels", "coverage", "uncertain", "explanation", "model"}:
        raise ValueError("Invalid evidence fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported evidence schema")
    if type(value["uncertain"]) is not bool:
        raise ValueError("Uncertainty must be Boolean")
    if not isinstance(value["coverage"], list) or set(value["coverage"]) != VISUAL_CATEGORIES:
        raise ValueError("Incomplete visual category coverage")
    if not isinstance(value["labels"], dict) or not set(value["labels"]) <= VISUAL_CATEGORIES:
        raise ValueError("Unknown evidence label")
    for confidence in value["labels"].values():
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and between zero and one")
    for key in ("model", "explanation"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 1000:
            raise ValueError(f"Invalid evidence {key}")
    return value


@dataclass(frozen=True)
class Decision:
    action: Action
    needs_review: bool = False
    reason: str = ""

    @property
    def mute_audio(self) -> bool:
        return self.action == Action.blur_mute


def resolve_action(policy: dict, evidence: dict | None, *, audio_override: bool | None = None,
                   action_override: str | None = None, legacy_mute: bool = False) -> Decision:
    """Resolve explicit scene intent first, then the union of category matches.

    Legacy/manual windows without evidence retain visual filtering; they are
    never assigned invented categories. Callers still enforce review approval.
    Unknown semantic evidence cannot auto-approve, even if its labels block.
    """
    policy = validate_policy(policy)
    if action_override is not None:
        action = Action(action_override)
        review = False
        reason = "Manual action override"
    elif evidence is None:
        action = Action.blur
        review = False
        reason = "Legacy / unclassified or manual scene"
    else:
        validate_evidence(evidence)
        matched = [policy["actions"][k] for k, score in evidence["labels"].items() if score >= 0.5]
        action = (Action.blur_mute if "blur_mute" in matched else
                  Action.blur if "blur" in matched else Action.allow)
        review = evidence["uncertain"] or any(0 < c < 0.5 for c in evidence["labels"].values())
        reason = evidence["explanation"]
    if action != Action.allow:
        mute = (audio_override if audio_override is not None else
                action == Action.blur_mute if action_override is not None else
                action == Action.blur_mute or policy["mute_blurred_audio"] or legacy_mute)
        action = Action.blur_mute if mute else Action.blur
    return Decision(action, review, reason)


def scene_set_fingerprint(scenes, policy: dict, render_settings: dict | None = None) -> str:
    """Fingerprint the approved effective set, not transient review metadata."""
    rows = []
    for scene in scenes:
        status = getattr(scene.status, "value", scene.status)
        if status == "approved":
            decision = scene_decision(scene, policy)
            rows.append((scene.id, scene.start_seconds, scene.end_seconds, decision.action.value))
    return hashlib.sha256(json.dumps({"rows": sorted(rows), "render": render_settings or {}},
                                     sort_keys=True).encode()).hexdigest()


def scene_decision(scene, policy: dict) -> Decision:
    """Shared ORM adapter for review previews and every render entry point."""
    raw = getattr(scene, "category_evidence", None)
    return resolve_action(
        policy, json.loads(raw) if raw else None,
        audio_override=getattr(scene, "audio_override", None),
        action_override=getattr(scene, "action_override", None),
        legacy_mute=bool(getattr(scene, "mute_audio", False)),
    )
