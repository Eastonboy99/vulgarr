"""NudeNet controls correspond to real labels and drive detection, not just UI."""
import asyncio
import copy
import json
from types import SimpleNamespace

from app.scenes.content_policy import (
    DEFAULT_POLICY, LEGACY_CATEGORIES, NUDENET_CATEGORIES, Action,
    nudenet_evidence, resolve_action, validate_policy,
)
from app.vision import classifier
from app.vision.nudenet_labels import NUDENET_LABELS, NUDENET_LABEL_DETAILS, category_id
from app.vision.scene_cluster import FrameScore


def test_exact_eighteen_upstream_labels_and_no_invented_garment_labels():
    assert len(NUDENET_LABELS) == len(set(NUDENET_LABELS)) == 18
    assert set(NUDENET_LABEL_DETAILS) == set(NUDENET_LABELS)
    assert not any('BIKINI' in label or 'CLEAVAGE' in label for label in NUDENET_LABELS)


def test_v1_policy_upgrade_preserves_preferences_without_guessing_clothing():
    old = {"schema_version": 1, "mute_blurred_audio": False,
           "actions": {k: DEFAULT_POLICY["actions"][k] for k in LEGACY_CATEGORIES}}
    old["actions"]["nudity.breasts"] = "allow"
    old["actions"]["clothing.bikini"] = "blur_mute"
    upgraded = validate_policy(old)
    assert upgraded["schema_version"] == 3
    assert upgraded["actions"][category_id("FEMALE_BREAST_EXPOSED")] == "allow"
    assert upgraded["actions"][category_id("FEMALE_BREAST_COVERED")] == "allow"
    assert upgraded["actions"]["clothing.bikini"] == "blur_mute"
    assert upgraded["mute_blurred_audio"] is False
    assert old["schema_version"] == 1  # No in-place mutation.


def test_every_actual_label_can_drive_proposals_and_effective_actions():
    for label in NUDENET_LABELS:
        detections = [{"class": label, "score": .91}]
        with classifier.detection_labels({label}):
            assert classifier.frame_confidence(detections) == .91
        with classifier.detection_labels(set()):
            assert classifier.frame_confidence(detections) == 0
        evidence = nudenet_evidence([FrameScore(0, .91, detections)], .3)
        assert set(evidence["coverage"]) == NUDENET_CATEGORIES
        assert evidence["labels"] == {category_id(label): .91}
        policy = copy.deepcopy(DEFAULT_POLICY)
        policy["actions"][category_id(label)] = "blur_mute"
        assert resolve_action(policy, evidence).action == Action.blur_mute
        policy["actions"][category_id(label)] = "allow"
        assert resolve_action(policy, evidence).action == Action.allow


def test_detector_threshold_is_respected_without_promoting_covered_to_bikini():
    observations = [FrameScore(0, .4, [{"class": "FEMALE_BREAST_COVERED", "score": .4}])]
    evidence = nudenet_evidence(observations, .3)
    policy = copy.deepcopy(DEFAULT_POLICY)
    policy["actions"][category_id("FEMALE_BREAST_COVERED")] = "blur"
    decision = resolve_action(policy, evidence, audio_override=False)
    assert decision.action == Action.blur and not decision.needs_review
    assert not any(k.startswith('clothing.') for k in evidence["labels"])
    assert nudenet_evidence([FrameScore(0, .9)], .3) is None


def test_concurrent_scans_cannot_leak_selected_classes():
    async def task(label):
        with classifier.detection_labels({label}):
            await asyncio.sleep(0)
            return classifier.frame_confidence([{"class": "FACE_FEMALE", "score": .8}])
    async def check():
        return await asyncio.gather(task("FACE_FEMALE"), task("FEET_EXPOSED"))
    assert asyncio.run(check()) == [.8, 0]
    assert classifier.frame_confidence([{"class": "FACE_FEMALE", "score": .8}]) == 0


def test_selected_class_cannot_resume_legacy_scalar_checkpoint(tmp_path):
    checkpoint = tmp_path/'scores.jsonl'
    checkpoint.write_text('{"t": 0, "c": 0.9}\n')
    with classifier.detection_labels({"FACE_FEMALE"}):
        assert classifier._load_resume_scores(checkpoint) == []
    detections = [{"class": "FACE_FEMALE", "score": .85}]
    checkpoint.write_text(json.dumps({"schema_version": 2, "t": 0, "c": 0, "detections": detections})+'\n')
    with classifier.detection_labels({"FACE_FEMALE"}):
        assert classifier._load_resume_scores(checkpoint)[0].confidence == .85
