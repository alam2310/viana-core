# UVH-26 training toolkit

Retrain the ITVA vehicle detector from the UVH-26 dataset. **Not required** for normal inference — production weights live in `models/v1/`.

## Data layout (host `data/`, gitignored)

```
data/
├── datasets/uvh26/          # Hugging Face download (raw UVH-26)
├── processed/yolo_format/   # COCO → YOLO conversion output
├── itva_phase1/             # balanced manifest (build script output)
└── outputs/
    ├── training/            # Ultralytics runs (best.pt)
    └── debug_pretrain/      # optional audit PNGs
```

Mount: `docker-compose.yml` maps `${VIANA_DATA_ROOT:-./data}` → `/data`. Scripts use repo-relative paths under `data/` (see `training/uvh/paths.py`).

## Workflow

Run inside the GPU container from repo root (`/app/ViAna`):

```bash
# 1. Download UVH-26 (once; see docs/ops/ENVIRONMENT_SETUP.md § optional dataset)
huggingface-cli download visual-layer/uvh26 --repo-type dataset \
  --local-dir data/datasets/uvh26 --local-dir-use-symlinks False

# 2. COCO → YOLO labels
python training/uvh/scripts/convert_coco_to_yolo.py

# 3. Taxonomy sanity check
python training/uvh/scripts/validate_taxonomy.py
python training/uvh/scripts/sync_taxonomy.py   # only if new raw labels found

# 4. Build balanced ITVA manifest + refresh itva_phase1.yaml
python training/uvh/scripts/build_itva_dataset.py

# 5. Train (dual GPU if available)
python training/uvh/train.py

# 6. Validate checkpoint
python training/uvh/validate.py

# 7. Promote weights manually after parity review
cp data/outputs/training/itva_phase1_1280p/weights/best.pt models/v1/itva_medium_1088p.pt
```

## Taxonomy

- JSON map: `training/uvh/taxonomy/vehicle_taxonomy.json`
- Human-readable table: `training/uvh/taxonomy/TAXONOMY.md`
- **Inference** taxonomy (production): `configs/classes.yaml` — do not confuse the two.

## Tests

```bash
pytest training/uvh/tests/ -q
```
