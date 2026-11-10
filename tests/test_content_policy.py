"""Policy tests use synthetic evidence, not claims of model accuracy."""
import copy
import json
from types import SimpleNamespace

import pytest

from app.scenes.content_policy import (
    Action, DEFAULT_POLICY, VISUAL_CATEGORIES, policy_fingerprint,
    resolve_action, scene_decision, validate_evidence, validate_policy,
)


def evidence(labels=None, **updates):
    return {"schema_version": 1, "labels": labels or {},
            "coverage": sorted(VISUAL_CATEGORIES), "uncertain": False,
            "explanation": "Synthetic test observation", "model": "mock-v1", **updates}


@pytest.mark.parametrize("labels,expected", [
    ({"clothing.low_cut": .95}, Action.allow),
    ({"clothing.chest_outline": .95}, Action.allow),
    ({"clothing.bikini": .95}, Action.blur_mute),
    ({"clothing.low_cut": .95, "clothing.bikini": .95}, Action.blur_mute),
    ({}, Action.allow),  # Ordinary bathroom dialogue has no blocking label.
])
def test_example_policy_and_multilabel_union(labels, expected):
    assert resolve_action(DEFAULT_POLICY, evidence(labels)).action == expected


def test_room_is_not_a_category():
    with pytest.raises(ValueError):
        validate_evidence(evidence({"bathroom": .99}))


def test_explicit_audio_override_wins_over_global_and_category_mute():
    assert resolve_action(DEFAULT_POLICY, evidence({"clothing.bikini": .9}), audio_override=False).action == Action.blur
    assert resolve_action(DEFAULT_POLICY, evidence({"clothing.low_cut": .9}), audio_override=True).action == Action.allow


def test_manual_action_and_unknown_evidence():
    uncertain = evidence({"clothing.bikini": .3}, uncertain=True)
    assert resolve_action(DEFAULT_POLICY, uncertain).needs_review
    assert resolve_action(DEFAULT_POLICY, uncertain, action_override="allow").action == Action.allow
    assert not resolve_action(DEFAULT_POLICY, uncertain, action_override="blur").needs_review


def test_policy_fingerprint_is_canonical_and_sensitive():
    policy = copy.deepcopy(DEFAULT_POLICY)
    assert policy_fingerprint(policy) == policy_fingerprint(json.loads(json.dumps(policy)))
    policy["actions"]["clothing.bikini"] = "allow"
    assert policy_fingerprint(policy) != policy_fingerprint(DEFAULT_POLICY)


@pytest.mark.parametrize("confidence", [float('nan'), float('inf'), -1, 2, True, "0.9"])
def test_invalid_confidence_is_not_safe_evidence(confidence):
    with pytest.raises(ValueError):
        validate_evidence(evidence({"clothing.bikini": confidence}))


def test_unavailable_categories_cannot_be_enabled():
    policy = copy.deepcopy(DEFAULT_POLICY)
    policy["actions"]["violence.graphic"] = "blur_mute"
    with pytest.raises(ValueError, match="backend"):
        validate_policy(policy)


def test_legacy_intent_and_explicit_reset():
    scene = SimpleNamespace(category_evidence=None, audio_override=False, mute_audio=False)
    assert scene_decision(scene, DEFAULT_POLICY).action == Action.blur
    scene.audio_override = None
    assert scene_decision(scene, DEFAULT_POLICY).action == Action.blur_mute


def test_missing_coverage_rejected_and_model_version_required():
    with pytest.raises(ValueError):
        validate_evidence(evidence(coverage=[]))
    with pytest.raises(ValueError):
        validate_evidence(evidence(model=""))
