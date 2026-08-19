"""UVH taxonomy mapping tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

UVH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UVH_ROOT))

from classifier import VehicleClassifier  # noqa: E402


@pytest.fixture
def classifier() -> VehicleClassifier:
    return VehicleClassifier(UVH_ROOT / "taxonomy" / "vehicle_taxonomy.json")


def test_tempo_traveller_mapping(classifier: VehicleClassifier) -> None:
    result = classifier.get_classification("Tempo-traveller")
    assert result["sub_class"] == "Mini Bus"
    assert result["class_type"] == "Light Fast"


def test_unknown_label(classifier: VehicleClassifier) -> None:
    result = classifier.get_classification("Flying-Saucer")
    assert result["category"] == "Unknown"
