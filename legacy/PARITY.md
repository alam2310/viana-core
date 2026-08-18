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

## Results (2026-08-19) — `hiv000001` **matched in-frame lines**

Same 180s window of `/data/raw/hiv000001.mp4` (`hiv000001_inframe.mp4`, 1920×1080, 2701 frames). Both pipelines used horizon `[0,648]–[1919,0]` and counting `[0,1079]–[1919,0]`.

### v2 conf 0.25 (same as legacy) + render

Job `job_parity_hiv000001_v2_c025`. Processed video: `/data/viana-outputs/parity_hiv000001/hiv000001_inframe_processed.mp4` (class names on boxes).

| Class | Legacy in / out / tot | v2@0.25 in / out / tot | v2@0.75 tot |
|-------|------------------------|-------------------------|-------------|
| Car | 1 / 29 / **30** | 2 / 27 / **29** | 12 |
| Jeep | 0 / 8 / **8** | 0 / 10 / **10** | 2 |
| Van | 0 / 3 / **3** | 0 / 3 / **3** | 3 |
| MiniBus | 0 / 1 / **1** | 0 / 1 / **1** | 1 |
| MTW | 0 / 42 / **42** | 1 / 42 / **43** | 32 |
| Auto | 0 / 17 / **17** | 0 / 15 / **15** | 8 |
| Bus | 1 / 4 / **5** | 1 / 4 / **5** | 0 |
| LCV | 0 / 4 / **4** | 0 / 4 / **4** | 0 |
| Cycle | 0 / 2 / **2** | 0 / 0 / **0** | 0 |
| Pedestrian | 4 / 6 / **10** | 38 / 9 / **47** | 13 |
| MCV | 0 / 3 / **3** | 0 / 3 / **3** | 1 |
| **All** | **125** | **160** | **72** |

Vehicle-only (exclude Pedestrian): legacy **115**, v2@0.25 **113**. Exact match on Bus, LCV, MCV, Van, MiniBus. Pedestrian (and missing Cycle) keep the run outside ±2% overall.

### Earlier v2 conf 0.75 (no render)

| Class | Legacy tot | v2@0.75 tot |
|-------|------------|-------------|
| (see table above) | 125 | 72 |

Van and MiniBus matched at 0.75; most vehicles were under-counted until conf was aligned.

## Results (2026-08-19) — `parity_golden.mp4` (geometry **not** matched)

Same 20s extract of `hiv00053_EDIT.mp4` (1920×1080). Lines and clip: `tests/viana/fixtures/PARITY_NOTES.md`.

### Per-class totals

| Class | Legacy (off-screen lines) | v2 CLI (clamped legacy-equivalent px) | v2 HTTP/UI (prescan lines) |
|-------|---------------------------|----------------------------------------|----------------------------|
| Pedestrian | 4 (in 3 / out 1) | 4 (in 2 / out 2) | 6 (in 6) |
| MTW | 0 | 1 (out) | 1 (in) |
| Auto | 3 (in 2 / out 1) | 0 | 0 |
| Heavy Truck | 3 (in 2 / out 1) | 0 | 0 |
| Jeep | 1 (out) | 0 | 0 |
| Bus | 1 (in) | 0 | 0 |
| **All classes** | **12** | **5** | **7** |

v2 CLI used in-frame diagonals `[0,648]–[1919,0]` / `[0,1079]–[1919,0]`. Legacy counting line y=1.15 / y=-0.15 is **not** the same segment. Annotated video: legacy `/tmp/legacy_reference.mp4` in container; UI run wrote `/data/viana-outputs/parity/parity_golden_processed.mp4`.

**Sign-off:** no — matched lines + matched conf (0.25) still miss ±2% overall because of Pedestrian (10 vs 47) and Cycle (2 vs 0). **Do not delete `legacy/`.**

## When to delete legacy/

- [x] Golden clip chosen and documented (`tests/viana/fixtures/PARITY_NOTES.md`)
- [x] Legacy vs v2 counts recorded (off-screen `parity_golden` **and** matched in-frame `hiv000001`; ±2% **not** met)
- [x] v2 HTTP path (same client as UI) run on a real-project extract (`parity_golden.mp4`)
- [ ] **Human sign-off** that count deltas are acceptable — **required before deleting `legacy/`**
- [ ] No open bugs referencing `legacy/inference/inference_engine.py`
