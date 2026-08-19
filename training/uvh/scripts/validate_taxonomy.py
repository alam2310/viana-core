"""Print taxonomy mapping validation table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import VehicleClassifier  # noqa: E402
from paths import TAXONOMY_JSON  # noqa: E402


def validate_mappings() -> None:
    classifier = VehicleClassifier(TAXONOMY_JSON)
    print(f"Loaded mapping from: {TAXONOMY_JSON}\n")

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

    print(f"{'RAW INPUT':<20} | {'CATEGORY':<12} | {'CLASS':<12} | {'SUB-CLASS':<15}")
    print("-" * 70)
    for raw_label in test_inputs:
        result = classifier.get_classification(raw_label)
        print(
            f"{raw_label:<20} | {result.get('category', 'N/A'):<12} | "
            f"{result.get('class_type', 'N/A'):<12} | {result.get('sub_class', 'N/A'):<15}"
        )

    tempo = classifier.get_classification("Tempo-traveller")
    if tempo["sub_class"] == "Mini Bus":
        print("\nPASS: Tempo-traveller -> Mini Bus")
    else:
        print(f"\nFAIL: Tempo-traveller -> {tempo['sub_class']}")


if __name__ == "__main__":
    validate_mappings()
