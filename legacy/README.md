# Legacy Codebase

**Discardable.** When `src/viana/` reaches parity and production sign-off, delete this entire `legacy/` directory.

Do not add new features here. Bug fixes only if needed for parity comparison.

## Layout

```
legacy/
├── blueprint.md           # Historical Phase 0–2 research log
├── PARITY.md              # Compare old vs new engine
├── README.md              # This file
├── docs/                  # Historical guides & taxonomy docs
├── configs/
│   └── vehicle_taxonomy.json   # UVH training label map
├── inference/             # Monolithic CV pipelines
│   └── inference_engine.py     ★ parity reference
├── training/              # Phase 1 model training & dataset tooling
├── scripts/               # Audit & taxonomy utilities
├── tests/
├── artifacts/             # runs/, debug_pretrain/, folderstructure.txt
└── weights/               # Unused experimental YOLO weights
```

## Active production paths (outside legacy)

| Concern | Location |
|---------|----------|
| New engine | `src/viana/` |
| API / jobs | `src/orchestrator/` |
| Class definitions (inference) | `configs/classes.yaml` |
| UVH label mapping (training only) | `legacy/configs/vehicle_taxonomy.json` |
| Production model | `models/v1/itva_medium_1088p.pt` |
| Pedestrian / training bases | `models/pretrained/yolo11l.pt`, `yolo11m.pt` |
| Living plan & status | `docs/PROJECT_PLAN.md`, `docs/PROJECT_STATUS.md` |

## Running legacy tools (from repo root, inside container)

```bash
python legacy/inference/inference_engine.py --video /data/.../clip.mp4 --out /tmp/legacy_out.mp4
python legacy/scripts/validate_taxonomy.py
python legacy/training/train.py   # retrain only if needed
```

## Tests

```bash
pytest legacy/tests/
pytest tests/viana/
```
