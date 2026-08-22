"""Phase 1 — classes.yaml and engine_defaults.yaml loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from viana.config.classes import load_class_taxonomy
from viana.config.defaults import load_engine_defaults
from viana.config.files import resolve_configs_dir
from viana.config.job import LineSegment, ViAnaTaskParameters


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.dump(payload), encoding="utf-8")
    return path


def test_load_committed_classes_yaml() -> None:
    """Load repo classes.yaml and match the v1 YOLO id table."""
    taxonomy = load_class_taxonomy()
    assert len(taxonomy.classes) == 15
    assert taxonomy.id_to_name()[0] == "Car"
    pedestrian = taxonomy.by_id(11)
    assert pedestrian.name == "Pedestrian"
    assert pedestrian.aggregate is True
    aggregatable = taxonomy.aggregatable()
    assert len(aggregatable) == 15
    assert any(item.id == 11 for item in aggregatable)
    assert taxonomy.by_name("Heavy Truck").id == 7


def test_load_committed_engine_defaults() -> None:
    """Load repo engine_defaults.yaml locked thresholds."""
    defaults = load_engine_defaults()
    assert defaults.detection.confidence_threshold == 0.75
    assert defaults.detection.imgsz == 1088
    assert defaults.classification.use_heuristic_truck_split is True
    assert defaults.models.vehicle == Path("models/v1/itva_medium_1088p.pt")
    assert defaults.models.pedestrian == Path("models/pretrained/yolo11l.pt")
    assert defaults.output.parent_dir == Path("/data/viana-outputs")
    assert defaults.pipeline.checkpoint_interval_frames == 500
    assert defaults.prescan.dark_frame_scan_sec == 4.0
    assert defaults.prescan.osd_probe_start_sec == 2.0
    assert defaults.prescan.osd_min_score == 20


def test_apply_task_overrides() -> None:
    """Job task_parameters override detection confidence and truck-split flag."""
    defaults = load_engine_defaults()
    params = ViAnaTaskParameters(
        horizon_line=LineSegment(start=(0, 0), end=(100, 0)),
        counting_line=LineSegment(start=(0, 200), end=(100, 200)),
        confidence_threshold=0.4,
        use_heuristic_truck_split=False,
    )
    merged = defaults.apply_task_overrides(params)
    assert merged.detection.confidence_threshold == 0.4
    assert merged.classification.use_heuristic_truck_split is False
    assert defaults.detection.confidence_threshold == 0.75
    assert defaults.classification.use_heuristic_truck_split is True


def test_missing_classes_file(tmp_path: Path) -> None:
    """Raise FileNotFoundError when classes.yaml is absent."""
    missing = tmp_path / "classes.yaml"
    with pytest.raises(FileNotFoundError):
        load_class_taxonomy(missing)


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    """Reject a YAML document that is not a mapping."""
    path = tmp_path / "classes.yaml"
    path.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_class_taxonomy(path)


def test_duplicate_class_ids_rejected(tmp_path: Path) -> None:
    """Reject two rows with the same YOLO id."""
    row = {
        "id": 0,
        "name": "Car",
        "category": "Passenger",
        "class_type": "Light Fast",
        "sub_class": "Car",
        "aggregate": True,
    }
    other = dict(row)
    other["name"] = "Clone"
    path = _write_yaml(tmp_path / "classes.yaml", {"classes": [row, other]})
    with pytest.raises(ValidationError, match="class ids must be unique"):
        load_class_taxonomy(path)


def test_unknown_class_key_rejected(tmp_path: Path) -> None:
    """Reject extra keys so taxonomy drift is caught early."""
    path = _write_yaml(
        tmp_path / "classes.yaml",
        {
            "classes": [
                {
                    "id": 0,
                    "name": "Car",
                    "category": "Passenger",
                    "class_type": "Light Fast",
                    "sub_class": "Car",
                    "aggregate": True,
                    "extra": "nope",
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        load_class_taxonomy(path)


def test_engine_defaults_unknown_section_rejected(tmp_path: Path) -> None:
    """Reject extra top-level keys in engine_defaults.yaml."""
    payload = load_engine_defaults().model_dump(mode="json")
    payload["unexpected"] = True
    path = _write_yaml(tmp_path / "engine_defaults.yaml", payload)
    with pytest.raises(ValidationError):
        load_engine_defaults(path)


def test_confidence_out_of_range_rejected(tmp_path: Path) -> None:
    """Reject detection confidence outside 0–1."""
    payload = load_engine_defaults().model_dump(mode="json")
    payload["detection"]["confidence_threshold"] = 1.5
    path = _write_yaml(tmp_path / "engine_defaults.yaml", payload)
    with pytest.raises(ValidationError):
        load_engine_defaults(path)


def test_resolve_configs_dir_explicit(tmp_path: Path) -> None:
    """Honor an explicit configs directory."""
    assert resolve_configs_dir(tmp_path) == tmp_path.resolve()


def test_unknown_class_id_raises() -> None:
    """Lookup of an unmapped YOLO id fails closed."""
    taxonomy = load_class_taxonomy()
    with pytest.raises(KeyError, match="unknown class id"):
        taxonomy.by_id(99)
