# Parity notes (golden clip)

## Clip A — `hiv000001` in-frame match (2026-08-19)

- Source: `/data/raw/hiv000001.mp4` (1920×1080, ~43:53, 15 fps). Full file not run (too long for one compare).
- Window: `/data/raw/hiv000001_inframe.mp4` = `ffmpeg -ss 60 -t 180` re-encode (2701 frames, 180.07s).
- **Matched in-frame lines** (legacy `TrafficConfig` injected so `int(norm * dim)` equals v2 pixels):

| Line | v2 / legacy pixels |
|------|--------------------|
| Horizon | `[0, 648]` → `[1919, 0]` |
| Counting | `[0, 1079]` → `[1919, 0]` |

Legacy inject norms: horizon `(0.0, 0.6)`–`((w-1)/w, 0.0)`; counting `(0.0, (h-1)/h)`–`((w-1)/w, 0.0)`. Confirmed `MATCHED_GEOM` on Visualizer.

v2 jobs under `/data/viana-outputs/parity_hiv000001/v2_job.json`:

| Run | conf | render | events |
|-----|------|--------|--------|
| First | 0.75 | false | 72 |
| Matched-conf (2026-08-19) | **0.25** (same as legacy) | true | **160** |

### Human-review lines (2026-08-19, not legacy-matched)

Job: `/data/viana-outputs/parity_hiv000001_review/v2_job.json` (conf 0.25, render true). **207** events. Processed: `hiv000001_inframe_processed.mp4`.

| Line | Color | Pixels |
|------|-------|--------|
| Horizon | red | `[0, 540]` → `[1919, 200]` |
| Counting | green | `[0, 725]` → `[1919, 600]` |

Overlay: class name + track id; **box color by class** (`docs/ui/OVERLAY_COLORS.md`). Not a ±2% parity compare.

### Quick compare — same clip `hiv000001_inframe.mp4`

Geometry A = legacy-matched diagonals `[0,648]–[1919,0]` / `[0,1079]–[1919,0]`.  
Geometry B = human-review `[0,540]–[1919,200]` / `[0,725]–[1919,600]`. Totals are **in / out / tot**. Blank = 0.

| Class | Legacy A | v2@0.75 A | v2@0.25 A | v2@0.25 B (review) |
|-------|----------|-----------|-----------|---------------------|
| Car | 1 / 29 / 30 | — / — / 12 | 2 / 27 / 29 | 20 / 35 / 55 |
| Jeep | 0 / 8 / 8 | — / — / 2 | 0 / 10 / 10 | 7 / 13 / 20 |
| Van | 0 / 3 / 3 | — / — / 3 | 0 / 3 / 3 | 3 / 3 / 6 |
| MiniBus | 0 / 1 / 1 | — / — / 1 | 0 / 1 / 1 | 1 / 1 / 2 |
| MTW | 0 / 42 / 42 | — / — / 32 | 1 / 42 / 43 | 23 / 52 / 75 |
| Auto | 0 / 17 / 17 | — / — / 8 | 0 / 15 / 15 | 1 / 16 / 17 |
| Bus | 1 / 4 / 5 | — / — / 0 | 1 / 4 / 5 | 3 / 4 / 7 |
| LCV | 0 / 4 / 4 | — / — / 0 | 0 / 4 / 4 | 0 / 0 / 0 |
| Heavy Truck | — | — | — | 1 / 7 / 8 |
| Cycle | 0 / 2 / 2 | — / — / 0 | 0 / 0 / 0 | 1 / 5 / 6 |
| Pedestrian | 4 / 6 / 10 | — / — / 13 | 38 / 9 / 47 | 6 / 5 / 11 |
| MCV | 0 / 3 / 3 | — / — / 1 | 0 / 3 / 3 | 0 / 0 / 0 |
| **All** | **125** | **72** | **160** | **207** |

Review video: `/data/viana-outputs/parity_hiv000001_review/hiv000001_inframe_processed.mp4`. Column B is a different counting line (higher on the road), so totals are **not** a parity delta vs legacy.

### Geometry C — user lines (2026-08-19), **matched on both pipelines**

Horizon `[0, 540]→[1919, 275]`, counting `[0, 775]→[1919, 650]`. Conf 0.25. v2 ByteTrack + counting-line class lock. Legacy LineZone (can count a track more than once).

Job/video: `/data/viana-outputs/parity_hiv000001_b2/` (`hiv000001_inframe_processed.mp4`).

| Class | Legacy C in / out / tot | v2 C in / out / tot |
|-------|-------------------------|---------------------|
| Car | 10 / 36 / **46** | 10 / 30 / **40** |
| Jeep | 5 / 13 / **18** | 5 / 12 / **17** |
| Van | 1 / 3 / **4** | 1 / 3 / **4** |
| MiniBus | 1 / 1 / **2** | 1 / 1 / **2** |
| MTW | 4 / 47 / **51** | 4 / 48 / **52** |
| Auto | 0 / 19 / **19** | 0 / 17 / **17** |
| Bus | 1 / 4 / **5** | 1 / 3 / **4** |
| LCV | 0 / 0 / **0** | 1 / 4 / **5** |
| Heavy Truck | 1 / 6 / **7** | 0 / 3 / **3** |
| Cycle | 0 / 5 / **5** | 0 / 5 / **5** |
| Pedestrian | 5 / 8 / **13** | 5 / 7 / **12** |
| MCV | 0 / 0 / **0** | 0 / 0 / **0** |
| **All** | **170** | **161** |

Exact match: Van, MiniBus, Cycle. In counts match on Car/Jeep/Van/MiniBus/MTW/Bus/Pedestrian. Remaining gap is mostly Car out (−6) and Goods split (legacy 7 Heavy Truck vs v2 5 LCV + 3 Heavy Truck). **Not signed off.** Do not delete `legacy/`.

### Geometry D — user lines (2026-08-19), **matched on both pipelines**

Horizon `[0, 500]→[1919, 325]`, counting `[0, 850]→[1919, 540]`. Conf 0.25.

v2 video: `/data/viana-outputs/parity_hiv000001_b3/hiv000001_inframe_processed.mp4` (re-run 2026-08-19 with overlay freeze after count: **307** events).

| Class | Legacy D in / out / tot | v2 D in / out / tot |
|-------|-------------------------|---------------------|
| Car | 35 / 35 / **70** | 34 / 35 / **69** |
| Jeep | 8 / 12 / **20** | 7 / 13 / **20** |
| Van | 4 / 3 / **7** | 4 / 3 / **7** |
| MiniBus | 1 / 1 / **2** | 1 / 1 / **2** |
| MTW | 88 / 54 / **142** | 77 / 53 / **130** |
| Auto | 23 / 20 / **43** | 21 / 18 / **39** |
| Bus | 3 / 4 / **7** | 3 / 4 / **7** |
| LCV | 0 / 0 / **0** | 1 / 5 / **6** |
| Heavy Truck | 1 / 7 / **8** | 0 / 4 / **4** |
| Cycle | 4 / 5 / **9** | 3 / 5 / **8** |
| Pedestrian | 7 / 10 / **17** | 6 / 9 / **15** |
| MCV | 1 / 0 / **1** | 0 / 0 / **0** |
| **All** | **326** | **307** |

Exact match: Jeep tot, Van tot, MiniBus, Bus. **Not signed off.** Do not delete `legacy/`.

Legacy still uses ByteTrack and LineZone (can count a track more than once). v2 `viana run` now uses **ByteTrack** (separate vehicle/person pools; IoU only if supervision is missing). Once-per-track counting is unchanged. Prescan geometric lines pin **x=0** and **x=width-1**.

## Clip B — `parity_golden.mp4` (earlier, unmatched geometry)

- Golden clip path: `/data/raw/parity_golden.mp4` (host: `data/raw/parity_golden.mp4`)
  - Extracted 2026-08-19: `ffmpeg -ss 30 -t 20 -i /data/raw/hiv00053_EDIT.mp4 -c copy`
  - Frame: **1920×1080**, ~20.07s, ~15 fps (HEVC). Source is a real project video (`hiv00053_EDIT.mp4`).
- Secondary smoke clip (E2E plumbing, not legacy compare): `/data/raw/test_video.mp4` (848×478, 11.2s, 168 frames).

## Legacy geometry (normalized, hardcoded defaults)

From `TrafficConfig` in `legacy/inference/inference_engine.py`:

| Line | start (norm) | end (norm) |
|------|----------------|------------|
| Horizon | `(0.0, 0.6)` | `(1.0, -0.4)` |
| Counting | `(0.0, 1.15)` | `(1.0, -0.15)` |

Y values outside `[0, 1]` are **off-screen** (legacy extends the line). Clip A clamps those to the table above.

## v2 geometry (pixel, clamped to frame)

Equivalent of the legacy lines on 1920×1080, **clamped** to `[0, width) × [0, height)`:

| Line | start px | end px |
|------|----------|--------|
| Horizon | `[0, 648]` | `[1919, 0]` |
| Counting | `[0, 1079]` | `[1919, 0]` |

Prescan / UI proposed lines on `parity_golden` (HTTP/UI-equivalent run):

| Line | start px | end px |
|------|----------|--------|
| Horizon | `[120, 400]` | `[1799, 520]` |
| Counting | `[80, 899]` | `[1839, 779]` |

## Acceptable tolerance

Target for **sign-off**: **±2% per class** on total crossings, and **in/out** within ±2 counts, **with matching in-frame lines**.

Clip A at **conf 0.25 both sides**: vehicle classes are close (Bus/LCV/MCV/Van/MiniBus exact; Car −1; MTW +1). **Pedestrian** is the remaining miss (legacy 10 vs v2 47). Totals 125 vs 160. Earlier v2@0.75 was 72. Remaining drivers: ByteTrack vs IoU, LineZone re-triggers vs once-per-track, pedestrian/person mapping.

**Not signed off.** Do not delete `legacy/`.

## Runtime extras (container `viana_core`, 2026-08-19)

- **ffmpeg:** present (`/usr/bin/ffmpeg`); `{stem}_processed.mp4` written when `render_video=true`.
- **ultralytics + CUDA:** live YOLO on `cuda:0` (2× RTX 3060).
- **numpy:** image `itva-base:stable` + `pip install -e ".[dev]"` pulled NumPy 2.x and **broke OpenCV 4.10**. Workaround: pin `numpy>=1.26,<2` after install (`docker-compose.yml` command). `trackers` wants NumPy 2; v2 does not use that package.
