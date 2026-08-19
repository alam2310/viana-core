"""Append missing raw UVH keys into vehicle_taxonomy.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import TAXONOMY_JSON  # noqa: E402

NEW_MAPPINGS = {
    "Others": {"category": "Goods", "class_type": "Light Fast", "sub_class": "Others"},
    "others": {"category": "Goods", "class_type": "Light Fast", "sub_class": "Others"},
    "Bicycle": {"category": "Passenger", "class_type": "Slow", "sub_class": "Cycle"},
    "bicycle": {"category": "Passenger", "class_type": "Slow", "sub_class": "Cycle"},
    "scooter": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW"},
    "bike": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW"},
    "motorcycle": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "MTW"},
    "auto": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "Auto"},
    "rickshaw": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "Auto"},
    "taxi": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "Car"},
    "tempo": {"category": "Passenger", "class_type": "Light Fast", "sub_class": "Mini Bus"},
    "tata-ace": {"category": "Goods", "class_type": "Light Fast", "sub_class": "LCV"},
}


def sync_taxonomy() -> None:
    if not TAXONOMY_JSON.is_file():
        print(f"Config file not found: {TAXONOMY_JSON}")
        sys.exit(1)

    with TAXONOMY_JSON.open(encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    print(f"Loaded taxonomy with {len(taxonomy)} keys.")

    added = 0
    for raw_key, mapping_data in NEW_MAPPINGS.items():
        key = raw_key.lower().strip()
        if key not in taxonomy:
            taxonomy[key] = mapping_data
            print(f"  Added: '{key}' -> {mapping_data['sub_class']}")
            added += 1

    if added:
        with TAXONOMY_JSON.open("w", encoding="utf-8") as handle:
            json.dump(taxonomy, handle, indent=2, sort_keys=True)
        print(f"Added {added} mappings to {TAXONOMY_JSON}")
    else:
        print("Taxonomy is already up to date.")


if __name__ == "__main__":
    sync_taxonomy()
