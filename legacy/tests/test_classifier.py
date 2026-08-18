import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "training"))

from utils.classifier import VehicleClassifier


def test_tempo_traveller_mapping():
    config_path = REPO_ROOT / "configs" / "vehicle_taxonomy.json"
    classifier = VehicleClassifier(str(config_path))
    result = classifier.get_classification("Tempo-traveller")
    assert result["sub_class"] == "Mini Bus"
    assert result["class_type"] == "Light Fast"


def test_unknown_label():
    config_path = REPO_ROOT / "configs" / "vehicle_taxonomy.json"
    classifier = VehicleClassifier(str(config_path))
    result = classifier.get_classification("Flying-Saucer")
    assert result["category"] == "Unknown"
