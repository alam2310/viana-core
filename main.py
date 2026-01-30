import sys
import os

# Ensure we can import from src/utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.classifier import VehicleClassifier

def validate_mappings():
    # 1. Initialize the classifier
    # Ensure the path points to where you actually saved the json file
    config_path = os.path.join("configs", "vehicle_taxonomy.json")
    
    try:
        classifier = VehicleClassifier(config_path)
        print(f"✅ Successfully loaded mapping from: {config_path}\n")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # 2. Define Test Cases (The Raw UVH-26 Labels + Some Edge Cases)
    test_inputs = [
        "Tempo-traveller",  # The specific case we fixed
        "Mini-bus",         # Should map to same as above
        "Bus",              # Should default to City Bus
        "MUV",              # Should merge into Car
        "Sedan",            # Should merge into Car
        "SUV",              # Should map to Jeep
        "Truck",            # Heavy Goods
        "LCV",              # Light Goods
        "Cycle",            # Slow Passenger
        "UFO_Object",       # Unknown object test
        "  van  "           # Whitespace test
    ]

    print(f"{'RAW INPUT':<20} | {'CATEGORY':<12} | {'CLASS':<12} | {'SUB-CLASS (Target)':<15}")
    print("-" * 70)

    # 3. Run Validation Loop
    for raw_label in test_inputs:
        result = classifier.get_classification(raw_label)
        
        # Extract values for clean printing
        cat = result.get('category', 'N/A')
        cls_type = result.get('class_type', 'N/A')
        sub_cls = result.get('sub_class', 'N/A')

        print(f"{raw_label:<20} | {cat:<12} | {cls_type:<12} | {sub_cls:<15}")

    # 4. specific Assertion for your Primary Requirement
    print("\n" + "="*30)
    print("CRITICAL LOGIC CHECK:")
    
    tempo_check = classifier.get_classification("Tempo-traveller")
    if tempo_check['sub_class'] == "Mini Bus":
         print("✅ PASS: 'Tempo-traveller' is correctly mapped to 'Mini Bus'")
    else:
         print(f"❌ FAIL: 'Tempo-traveller' is mapped to {tempo_check['sub_class']}")

if __name__ == "__main__":
    validate_mappings()