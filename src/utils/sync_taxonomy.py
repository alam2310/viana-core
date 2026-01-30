import json
import os
import sys

# Define the Path to your Single Source of Truth
CONFIG_PATH = os.path.join("configs", "vehicle_taxonomy.json")

def sync_taxonomy():
    """
    Updates the vehicle_taxonomy.json file with new raw keys found during the UVH-26 audit.
    This ensures that the JSON config remains the Single Source of Truth.
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    try:
        with open(CONFIG_PATH, 'r') as f:
            taxonomy = json.load(f)
        print(f"✅ Loaded existing taxonomy with {len(taxonomy)} keys.")
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        sys.exit(1)

    # 3. Define New Raw Keys to Sync
    new_mappings = {
        # --- MISSING KEYS FIXED HERE ---
        "Others": {
            "category": "Goods", # Defaulting to Goods/Utility for 'Others'
            "class_type": "Light Fast",
            "sub_class": "Others"
        },
        "others": {
            "category": "Goods",
            "class_type": "Light Fast",
            "sub_class": "Others"
        },
        
        # --- Previous Fixes (Kept for safety) ---
        "Bicycle": { "category": "Passenger", "class_type": "Slow", "sub_class": "Cycle" },
        "bicycle": { "category": "Passenger", "class_type": "Slow", "sub_class": "Cycle" },
        "scooter": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW" },
        "bike": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW" },
        "motorcycle": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW" },
        "auto": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "Auto" },
        "rickshaw": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "Auto" },
        "taxi": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "Car" },
        "tempo": { "category": "Passenger", "class_type": "Light Fast", "sub_class": "Mini Bus" },
        "tata-ace": { "category": "Goods", "class_type": "Light Fast", "sub_class": "LCV" }
    }

    # 4. Sync Logic
    added_count = 0
    for raw_key, mapping_data in new_mappings.items():
        key = raw_key.lower().strip() # Normalize to lowercase for the JSON key
        
        if key not in taxonomy:
            taxonomy[key] = mapping_data
            print(f"   ➕ Added missing key: '{key}' -> {mapping_data['sub_class']}")
            added_count += 1

    # 5. Save
    if added_count > 0:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(taxonomy, f, indent=2, sort_keys=True)
        print(f"\n✅ Successfully added {added_count} new mappings to {CONFIG_PATH}")
    else:
        print("\n✅ Taxonomy is already up to date.")

if __name__ == "__main__":
    sync_taxonomy()