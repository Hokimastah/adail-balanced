from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MedicalDatasetPreset:
    """Metadata for the medical benchmarks described in the ADAIL proposal."""

    name: str
    num_classes: int
    default_scenarios: tuple[str, ...]
    ordinal: bool
    class_names: tuple[str, ...]
    description: str


PRESETS: dict[str, MedicalDatasetPreset] = {
    "cch5000": MedicalDatasetPreset(
        name="CCH5000",
        num_classes=8,
        default_scenarios=("4-1", "4-2"),
        ordinal=False,
        class_names=(
            "stroma",
            "tumor",
            "complex",
            "lympho",
            "debris",
            "mucosa",
            "adiposa",
            "empty_background",
        ),
        description=(
            "Histology benchmark in the proposal: 8 colorectal tissue morphology classes, "
            "balanced at about 625 images per class; used for 4-1 and 4-2 CIL splits."
        ),
    ),
    "ham10000": MedicalDatasetPreset(
        name="HAM10000",
        num_classes=7,
        default_scenarios=("3-1", "3-2"),
        ordinal=False,
        class_names=(
            "akiec",
            "bcc",
            "bkl",
            "df",
            "mel",
            "nv",
            "vasc",
        ),
        description=(
            "Dermatoscopy benchmark in the proposal: 10,015 images with severe long-tail "
            "class imbalance; used for 3-1 and 3-2 CIL splits."
        ),
    ),
    "eyepacs": MedicalDatasetPreset(
        name="EyePACS",
        num_classes=5,
        default_scenarios=("3-1", "3-2"),
        ordinal=True,
        class_names=("no_dr", "mild", "moderate", "severe", "proliferative"),
        description=(
            "Diabetic retinopathy fundus benchmark in the proposal: 5 severity stages "
            "with ordinal structure from No DR to Proliferative DR."
        ),
    ),
}


def get_preset(name: str) -> MedicalDatasetPreset:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "cch5000": "cch5000",
        "crc5000": "cch5000",
        "ham10000": "ham10000",
        "ham": "ham10000",
        "eyepacs": "eyepacs",
        "eye": "eyepacs",
    }
    if key not in aliases:
        valid = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available presets: {valid}")
    return PRESETS[aliases[key]]
