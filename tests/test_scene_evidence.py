"""Raw NudeNet evidence must not impersonate semantic clothing coverage."""
import json

from app.vision.classifier import _load_resume_scores


def test_v2_checkpoint_retains_per_class_evidence(tmp_path):
    checkpoint = tmp_path/"scores.jsonl"
    detections = [{"class": "FEMALE_BREAST_COVERED", "score": .9, "box": [1, 2, 3, 4]}]
    checkpoint.write_text(json.dumps({"schema_version": 2, "t": 0, "c": 0, "detections": detections}) + "\n")
    score = _load_resume_scores(checkpoint)[0]
    assert score.confidence == 0
    assert score.detections == detections
    assert not any(d["class"].startswith("clothing.") for d in score.detections)


def test_legacy_scalar_checkpoint_has_no_invented_raw_evidence(tmp_path):
    checkpoint = tmp_path/"scores.jsonl"
    checkpoint.write_text('{"t": 0, "c": 0.95}\n')
    score = _load_resume_scores(checkpoint)[0]
    assert score.confidence == .95
    assert score.detections is None
