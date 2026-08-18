# Parity Testing: Legacy vs ViAna v2

Compare `legacy/inference/inference_engine.py` against `python -m viana run` on the same clip before deleting `legacy/`.

## Reference command (legacy)

```bash
cd /app/ViAna

python legacy/inference/inference_engine.py \
  --video /data/path/to/reference_clip.mp4 \
  --model_a models/v1/itva_medium_1088p.pt \
  --model_b models/pretrained/yolo11l.pt \
  --out /tmp/legacy_reference.mp4
```

Legacy uses hardcoded geometry in the script. For fair comparison, note the `TrafficConfig` values in `inference_engine.py` and convert to pixel coords for the v2 job config, **clamped to frame bounds** (v2 does not use off-screen line extension).

## Target command (v2 — when Phase 3+ is complete)

```bash
python -m viana run --config /tmp/parity_job.json
```

## What to compare

| Metric | Legacy | v2 |
|--------|--------|-----|
| Total crossings per class | console totals | `{stem}_events.csv` row counts |
| Direction in/out | `counts_in` / `counts_out` | `direction` column in events CSV |
| Annotated video | stdout path | `{stem}_processed.mp4` |

Tolerance: document acceptable delta (e.g. ±2% per class) in `tests/viana/fixtures/PARITY_NOTES.md` when golden clip is chosen.

## Geometry note

Legacy counting line uses normalized coords that extend off-screen (e.g. y=1.15). v2 requires endpoints inside the frame — expect small count differences until lines are calibrated equivalently within bounds.

## When to delete legacy/

- [ ] Golden clip parity signed off
- [ ] v2 prescan + canvas workflow validated on real project videos
- [ ] No open bugs referencing `legacy/inference/inference_engine.py`
