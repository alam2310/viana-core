"""Legacy vehicle taxonomy validation script."""

import sys
import os
from pathlib import Path

# Repo root (legacy/scripts → ../../)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "training"))

from utils.classifier import VehicleClassifier


def validate_mappings() -> None:
    """Print taxonomy mapping validation table."""
    config_path = REPO_ROOT / "legacy" / "configs" / "vehicle_taxonomy.json"

    try:
        classifier = VehicleClassifier(str(config_path))
        print(f"✅ Successfully loaded mapping from: {config_path}\n")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    test_inputs = [
        "Tempo-traveller",
        "Mini-bus",
        "Bus",
        "MUV",
        "Sedan",
        "SUV",
        "Truck",
        "LCV",
        "Cycle",
        "UFO_Object",
        "  van  ",
    ]

    print(f"{'RAW INPUT':<20} | {'CATEGORY':<12} | {'CLASS':<12} | {'SUB-CLASS (Target)':<15}")
    print("-" * 70)

    for raw_label in test_inputs:
        result = classifier.get_classification(raw_label)
        cat = result.get("category", "N/A")
        cls_type = result.get("class_type", "N/A")
        sub_cls = result.get("sub_class", "N/A")
        print(f"{raw_label:<20} | {cat:<12} | {cls_type:<12} | {sub_cls:<15}")

    print("\n" + "=" * 30)
    print("CRITICAL LOGIC CHECK:")
    tempo_check = classifier.get_classification("Tempo-traveller")
    if tempo_check["sub_class"] == "Mini Bus":
        print("✅ PASS: 'Tempo-traveller' is correctly mapped to 'Mini Bus'")
    else:
        print(f"❌ FAIL: 'Tempo-traveller' is mapped to {tempo_check['sub_class']}")


if __name__ == "__main__":
    validate_mappings()
