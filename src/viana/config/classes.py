"""Class taxonomy loaded from ``configs/classes.yaml``."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from viana.config.files import resolve_configs_dir
from viana.config.yaml_io import load_yaml_mapping


class VehicleClass(BaseModel):
    """One YOLO class id mapped to the reporting hierarchy."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    class_type: str = Field(min_length=1)
    sub_class: str = Field(min_length=1)
    aggregate: bool


class ClassTaxonomy(BaseModel):
    """Full inference-time class table (single source of truth)."""

    model_config = ConfigDict(extra="forbid")

    classes: list[VehicleClass] = Field(min_length=1)
    _by_id: dict[int, VehicleClass] = PrivateAttr(default_factory=dict)
    _by_name: dict[str, VehicleClass] = PrivateAttr(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def unique_ids_and_names(cls, classes: list[VehicleClass]) -> list[VehicleClass]:
        """Reject duplicate YOLO ids or display names."""
        ids = [item.id for item in classes]
        names = [item.name for item in classes]
        if len(ids) != len(set(ids)):
            raise ValueError("class ids must be unique")
        if len(names) != len(set(names)):
            raise ValueError("class names must be unique")
        return classes

    def model_post_init(self, __context: object) -> None:
        """Index classes by YOLO id and display name."""
        self._by_id = {item.id: item for item in self.classes}
        self._by_name = {item.name: item for item in self.classes}

    def by_id(self, class_id: int) -> VehicleClass:
        """Return the class row for a YOLO id."""
        try:
            return self._by_id[class_id]
        except KeyError as exc:
            raise KeyError(f"unknown class id: {class_id}") from exc

    def by_name(self, name: str) -> VehicleClass:
        """Return the class row for a display name."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown class name: {name}") from exc

    def id_to_name(self) -> dict[int, str]:
        """Return YOLO id → display name (legacy ``class_names`` shape)."""
        return {item.id: item.name for item in self.classes}

    def aggregatable(self) -> list[VehicleClass]:
        """Classes included in ``{stem}_15min.csv`` (``aggregate: true``)."""
        return [item for item in self.classes if item.aggregate]


def load_class_taxonomy(path: Path | None = None) -> ClassTaxonomy:
    """Load and validate ``classes.yaml``.

    Args:
        path: Explicit file path. When omitted, ``<configs>/classes.yaml`` is used.
    """
    yaml_path = path if path is not None else resolve_configs_dir() / "classes.yaml"
    payload = load_yaml_mapping(yaml_path)
    return ClassTaxonomy.model_validate(payload)
