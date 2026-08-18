# Legacy Codebase

**Discardable.** When `src/viana/` reaches parity and production sign-off, delete this entire `legacy/` directory.

Do not add new features here. Bug fixes only if needed for parity comparison.

## Layout

```
legacy/
├── PARITY.md              # How to compare old vs new engine
├── inference/             # Monolithic CV pipelines (reference)
│   └── inference_engine.py   ★ parity reference
├── training/              # Phase 1 model training & dataset tooling
│   ├── train.py
│   └── utils/
├── scripts/               # One-off dataset / taxonomy utilities
│   ├── validate_taxonomy.py  (was root main.py)
│   ├── audit_dataset.py
│   ├── audit_uvh26.py
│   └── convert_and_audit.py
├── tests/                 # Tests for legacy taxonomy classifier
│   └── test_classifier.py
├── artifacts/             # Old outputs, debug images, folder snapshots
│   ├── runs/
│   ├── debug_pretrain/
│   └── folderstructure.txt
└── weights/               # Unused experimental YOLO weights
    ├── yolo11s.pt
    └── yolo26n.pt
```

## Active production paths (outside legacy)

| Concern | Location |
|---------|----------|
| New engine | `src/viana/` |
| API / jobs | `src/orchestrator/` |
| Class definitions (inference) | `configs/classes.yaml` |
| UVH label mapping (training only) | `configs/vehicle_taxonomy.json` |
| Production model | `models/v1/itva_medium_1088p.pt` |
| Pedestrian / training bases | `models/pretrained/yolo11l.pt`, `yolo11m.pt` |

## Running legacy tools (from repo root, inside container)

```bash
# Parity reference pipeline
python legacy/inference/inference_engine.py --video /data/.../clip.mp4 --out /tmp/legacy_out.mp4

# Taxonomy validation
python legacy/scripts/validate_taxonomy.py

# Retrain (Phase 1 — only if needed)
python legacy/training/train.py
```

## Tests

```bash
pytest legacy/tests/   # vehicle_taxonomy.json mapping only
pytest tests/viana/    # new engine (active)
```
