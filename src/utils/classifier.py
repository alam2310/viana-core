import json
import os

class VehicleClassifier:
    def __init__(self, config_path="configs/vehicle_taxonomy.json"):
        self.mapping = self._load_mapping(config_path)

    def _load_mapping(self, path):
        """Loads the JSON mapping file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mapping file not found at {path}")
        
        with open(path, 'r') as f:
            return json.load(f)

    def get_classification(self, raw_label):
        """
        Returns the 3-level classification for a given raw label.
        Returns 'Unknown' category if label is not found.
        """
        # Normalize input to match JSON keys (lowercase, strip whitespace)
        key = raw_label.lower().strip()
        
        # Default fallback
        fallback = {
            "category": "Unknown",
            "class_type": "Unknown",
            "sub_class": raw_label
        }
        
        return self.mapping.get(key, fallback)

# --- Usage Example ---
if __name__ == "__main__":
    classifier = VehicleClassifier()
    
    # Example 1: The Tempo Traveller (The key requirement)
    # Even though UVH sees "Tempo-traveller", we get "Mini Bus"
    result = classifier.get_classification("Tempo-traveller")
    print(f"Input: Tempo-traveller -> {result}") 
    # Output: {'category': 'Passenger', 'class_type': 'Light Fast', 'sub_class': 'Mini Bus'}

    # Example 2: MUV
    result = classifier.get_classification("MUV")
    print(f"Input: MUV -> {result}")
    # Output: {'category': 'Passenger', 'class_type': 'Light Fast', 'sub_class': 'Car'}