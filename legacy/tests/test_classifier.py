import pytest
import os
from src.utils.classifier import VehicleClassifier

def test_tempo_traveller_mapping():
    # Setup
    config_path = os.path.join(os.path.dirname(__file__), '../configs/vehicle_taxonomy.json')
    classifier = VehicleClassifier(config_path)
    
    # Action
    result = classifier.get_classification("Tempo-traveller")
    
    # Assert
    assert result['sub_class'] == "Mini Bus"
    assert result['class_type'] == "Light Fast"

def test_unknown_label():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/vehicle_taxonomy.json')
    classifier = VehicleClassifier(config_path)
    
    result = classifier.get_classification("Flying-Saucer")
    assert result['category'] == "Unknown"