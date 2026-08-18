# Model weights

Production inference uses paths from `configs/engine_defaults.yaml`.

| Path | Role | Used in v0.1 |
|------|------|--------------|
| `models/v1/itva_medium_1088p.pt` | Production vehicle detector/classifier (1088p ITVA) | **Yes** |
| `models/pretrained/yolo11l.pt` | Pedestrian detector (pretrained YOLO11-L) | **Yes** |
| `models/public/weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt` | UVH-26 research / training checkpoint | **No** (legacy training reference) |

## Notes

- **v1/** — canonical production weights for the v2 engine (`ViAna_Moving`).
- **pretrained/** — third-party YOLO weights; not fine-tuned on ITVA data.
- **public/** — historical UVH-26 experiment weights kept for training parity and audit. Do not point `engine_defaults.yaml` here unless explicitly migrating models.

Large `.pt` files may be tracked via Git LFS. If weights are missing locally, obtain them from the project maintainer or retrain using `legacy/training/` (historical).
