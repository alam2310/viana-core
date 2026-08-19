# ITVA research log (pre-v2)

Distilled from the legacy Phase 0–2 era. **Current implementation:** `src/viana/`, `docs/PROJECT_PLAN.md`, `docs/PROJECT_STATUS.md`.

Hardware reference: i7-12700F, 32 GB RAM, 2× RTX 3060 (12 GB), host Ubuntu 24.04, container Ubuntu 22.04.

---

## Environment decisions (what worked)

| Decision | Rationale |
|----------|-----------|
| Docker-first dev | Avoid host dependency drift; reproducible CUDA/OpenCV |
| Container Ubuntu 22.04 + CUDA 12.4 | Stable PyTorch cu124 and OpenCV 4.10 compile |
| Custom OpenCV with CUDA | `apt` OpenCV lacks GPU backends |
| PyTorch from cu124 wheel index | Avoid driver/toolkit version mismatch |
| FFmpeg for video I/O / encode | GPU decode via NVCUVID abandoned; NVENC for small MP4 |
| Dual-GPU inference | Vehicle model on `cuda:0`, pedestrian on `cuda:1` |
| NumPy &lt; 2 in runtime | OpenCV 4.10 breaks on NumPy 2.x |
| `trackers` ByteTrack, install `--no-deps` | Package wants NumPy 2; pin separately |

---

## Failed attempts (do not repeat)

| Attempt | Why it failed |
|---------|----------------|
| Ubuntu 24.04 + CUDA 13.1 base image | Breaking API changes; OpenCV source incompatible |
| OpenCV compile with `NVCUVID=ON` | `nvcuvid.h` deprecated/removed in CUDA 12.x headless builds |
| `cv2.VideoWriter` for annotated output | Huge uncompressed files; replaced by FFmpeg `hevc_nvenc` pipe |
| YOLO11-Nano/Small pedestrian sidecar | False positives on riders and vehicle parts; upgraded to YOLO11-Medium on GPU 1 |
| Static pixel-area truck thresholds | Perspective makes distant trucks look like MCV; v2 uses depth-normalized area |
| Inline 15-minute CSV in GPU loop | Never completed in legacy; v2 uses events CSV + separate `viana aggregate` (ADR 001) |

---

## Training (UVH-26 → ITVA)

- **Imbalance:** Mini Bus &lt; 1% vs MTW ~47% in raw UVH-26.
- **Fix:** Config-driven taxonomy (`training/uvh/taxonomy/vehicle_taxonomy.json`) + manifest oversampling (Mini Bus / Van ×20, LCV ×5).
- **Model pivot:** YOLO11-Large training too slow → **YOLO11-Medium** at **1088p**; mAP@50 &gt; 0.92 by epoch 7 in original run.
- **Production weights:** `models/v1/itva_medium_1088p.pt`. Retrain procedure: `training/README.md`.

---

## Inference pipeline evolution (now in `src/viana/`)

1. Dual YOLO detect + IoA suppression (person inside vehicle box &gt; 30%).
2. ByteTrack (separate vehicle / pedestrian pools in v2).
3. Horizon filter (ignore boxes above horizon line).
4. N-frame majority class lock near counting line + heuristic truck split (Heavy Truck → LCV / MCV / Trailer by area and aspect).
5. Once-per-track line crossing (legacy LineZone could re-count).
6. EasyOCR for burned-in time/location (v2: `viana prescan` + `time_map`; aggregate is separate).
7. HEVC NVENC annotated video (cq 42 in v2; legacy used cq 38).

**Parity (2026-08-19):** Geometry D on `hiv000001_inframe.mp4` — legacy 326 vs v2 307 events; overlay review **human go**. Legacy tree removed; counts recorded in `tests/viana/fixtures/PARITY_NOTES.md`.

---

## Lessons learned

- **Rule-based geometry alone fails** when depth changes — normalize box area using horizon-relative Y before truck/LCV splits.
- **Tracker memory matters** — increase buffer for occlusion; use `counted_ids` / freeze so jitter does not double-count.
- **Total counts are insufficient** for operations — need wall-clock bins (15-min CSV); legacy only logged interval boundaries, never exported CSV.
- **Config over code for taxonomy** — raw UVH labels map via JSON; inference taxonomy is `configs/classes.yaml`.

---

## Deferred ideas (not in v0.1)

- **15-minute CSV product** — legacy Action 2.7 never shipped; v2 has `viana aggregate` but needs prescan UI + wall-clock wiring.
- **NVIDIA DALI** — GPU video decoding instead of OpenCV/FFmpeg CPU read.
- **Python multiprocessing** — one worker process per GPU for batch farm (orchestrator already queues 2 jobs).
- **Perspective “reference box” auto-calibration** — NEAR_SCALE / FAR_SCALE from traffic samples.
- **Secondary attribute classifier** — ResNet/YOLO-cls for Taxi (yellow plate), axle count, Intercity Bus vs City Bus.
- **Flow-based auto counting line** — PCA on trajectories for diagonal roads / camera bump recovery.
- **Trailer as first-class YOLO class** — currently heuristic from aspect ratio on Heavy Truck votes.
- **GPU CI** — no GPU in GitHub Actions; manual/container verification only.
- **Bake `trackers` + numpy pin into image** — compose still pip-installs on container start.
- **Browser E2E automation** — Playwright click-through of Next.js (HTTP client E2E exists).
- **Pause/resume UX** — needs jobs longer than 500-frame checkpoint interval.
- **ViAnaNP parked extraction** — separate product line (see `docs/project_context.md`).

---

## Architecture decision cross-reference

Overlaps with ADRs: event-sourced analytics (001), backend job management (002). This log is narrative context only.
