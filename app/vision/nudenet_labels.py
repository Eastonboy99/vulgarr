"""NudeNet 3.4.2's exact 18 detector labels, in upstream model order.

Source: https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py
Covered body parts are observations, not clothing-type or intent classifiers.
Keep this list shared by policy, UI and scoring; do not invent semantic labels.
"""
NUDENET_LABELS = (
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)
DEFAULT_BLOCKED_LABELS = frozenset({
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
})
NUDENET_LABEL_DETAILS = {
    "FEMALE_GENITALIA_COVERED": ("Female genitalia · covered", "covered"),
    "FACE_FEMALE": ("Female face", "other"),
    "BUTTOCKS_EXPOSED": ("Buttocks · exposed", "exposed"),
    "FEMALE_BREAST_EXPOSED": ("Female breast · exposed", "exposed"),
    "FEMALE_GENITALIA_EXPOSED": ("Female genitalia · exposed", "exposed"),
    "MALE_BREAST_EXPOSED": ("Male breast · exposed", "exposed"),
    "ANUS_EXPOSED": ("Anus · exposed", "exposed"),
    "FEET_EXPOSED": ("Feet · exposed", "other"),
    "BELLY_COVERED": ("Belly · covered", "covered"),
    "FEET_COVERED": ("Feet · covered", "other"),
    "ARMPITS_COVERED": ("Armpits · covered", "other"),
    "ARMPITS_EXPOSED": ("Armpits · exposed", "other"),
    "FACE_MALE": ("Male face", "other"),
    "BELLY_EXPOSED": ("Belly · exposed", "other"),
    "MALE_GENITALIA_EXPOSED": ("Male genitalia · exposed", "exposed"),
    "ANUS_COVERED": ("Anus · covered", "covered"),
    "FEMALE_BREAST_COVERED": ("Female breast · covered", "covered"),
    "BUTTOCKS_COVERED": ("Buttocks · covered", "covered"),
}
NUDENET_GROUPS = (
    ("exposed", "Exposed anatomy", "Direct exposed-anatomy detections."),
    ("covered", "Covered anatomy", "Covered body parts—not specific garments or levels of modesty."),
    ("other", "Other detector labels", "Faces, belly, feet and armpits. Usually left allowed."),
)


def category_id(label: str) -> str:
    """Namespace detector IDs separately from legacy semantic categories."""
    return f"nudenet.{label}"
