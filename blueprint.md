# 🗺️ Project Blueprint: Indian Traffic Video Analytics (ITVA)
**Objective:** End-to-end offline pipeline for high-accuracy vehicle classification and counting.
**Hardware:** i7-12700F | 32GB DDR5 | 2x RTX 3060 (12GB) | Host: Ubuntu 24.04 | Container: Ubuntu 22.04
**Target Classes:** 2-Wheelers, Cars, Auto Rickshaws, Trucks, Medium Vehicles, Light Vehicles, Heavy Vehicles.

---

## 🚦 Project Master Status
- **Current Phase:** Phase 2: High-Accuracy Offline Engine (Logic Remediation & Stabilization)
- **Overall Progress:** 70% (Inference, Tracking, and Export Stable; Core Logic Fixes Next)
- **Last Updated:** 2026-02-26

---

## 🛠️ Phase Registry

### Phase 0: Environment Foundation 
**Status:** ✅ COMPLETE
- [x] Install NVIDIA Driver 590+ on Host.
- [x] Configure Docker with NVIDIA Container Toolkit 1.18+.
- [x] **Pivot:** Build "Safe Landing" Base Image (Ubuntu 22.04 + CUDA 12.4 + OpenCV 4.10).
    - *Verified:* CUDA 12.4 support, Dual-GPU visibility, OpenCV CUDA modules active.
    - *Decision:* Abandoned `NVCUVID` compilation in favor of NVIDIA DALI / FFmpeg for decoding.
- [x] **Verification:** Validated GPU load with PyTorch (Matrix Mul) and OpenCV (GpuMat Resize).
- [x] **Refinement:** Established "Golden Master" v3.0 Environment Guide.
    - *Implemented:* `docker-compose.yml` for standardized workflow.
    - *Fixed:* Dataset Symbolic Link bridge (Host `/root/Work` <-> Container `/app`) baked into Dockerfile.

#### ⚠️ History of Failed Attempts (Phase 0)
*These approaches were attempted and discarded to ensure stability.*
- [❌] **FAILED:** Build "Bleeding Edge" Image (Ubuntu 24.04 + CUDA 13.1). 
    - *Reason:* Too many breaking API changes in CUDA 13.1; incompatible with current OpenCV source.
- [❌] **FAILED:** Compile OpenCV 4.10 with `NVCUVID=ON` inside Docker.
    - *Reason:* NVIDIA deprecated headers (nvcuvid.h) in CUDA 12.x. "Headless" build environments cannot reliably satisfy linker checks for runtime drivers without fragile hacks.

### Phase 1: Model & Dataset Strategy
**Status:** ✅ COMPLETE
- [x] **Action 1.1: Audit UVH-26 Dataset.**
    - *Outcome:* Identified critical imbalance (Mini Bus < 1% vs MTW 47%).
    - *Artifact:* `src/utils/dataset_auditor.py`.
- [x] **Action 1.2: Design Taxonomy & Logic.**
    - *Strategy:* **Config-Driven Architecture**. Decoupled logic from code using `vehicle_taxonomy.json`.
    - *Outcome:* Normalized chaotic labels (Tempo, Tata Ace) into structured Target IDs.
- [x] **Action 1.3: Engineer Balanced Dataset.**
    - *Strategy:* **Manifest Oversampling**. Multiplied rare classes (Mini Bus 20x, LCV 5x) in the training manifest.
    - *Outcome:* `itva_phase1` dataset created. Effective balance achieved without synthetic data.
- [x] **Action 1.4: Train Model (The "Brain").**
    - [x] **Pivot to Medium:** Switched from YOLO11-Large to **YOLO11-Medium** to resolve 5-day training bottleneck.
    - [x] **High-Res Strategy:** Trained at **1088p** (1088px) to secure small-object recall.
    - [x] **Outcome:** Achieved **mAP@50 > 0.92** and **Recall > 0.88** by Epoch 7.
    - [x] **Validation:** Confirmed "Mini Bus" class is generalizing (not purely overfitting).

### Phase 2: High-Accuracy Offline Engine
**Status:** 🚀 IN PROGRESS
- [x] **Action 2.1: The "Ensemble" Inference Engine.**
    - *Goal:* Run two models simultaneously to cover Vehicles + Pedestrians.
    - **Iteration Log (The Accuracy Climb):**
        - [x] **Attempt 1 (Nano Sidecar):** Used `yolo11n.pt` for pedestrians. *Result:* High speed, but false positives on Motorbike Riders and Car parts. Low confidence (0.35).
        - [x] **Attempt 2 (Rider Suppression):** Implemented IoA Logic (`Intersection / Person_Area > 0.3`) to suppress people overlapping with 2-Wheelers. *Result:* Fixed "Rider" issue, but "Car Side" false positives remained.
        - [x] **Attempt 3 (Small + Universal Suppression):** Upgraded to `yolo11s.pt` and checked overlap against *ALL* vehicles. *Result:* Better, but confidence still borderline (0.40 - 0.65).
        - [x] **Attempt 4 (Dual-GPU Isolation):** **Final Strategy.** Pushed Model A (Vehicles) to `cuda:0` and Model B (Pedestrians, **YOLO11-Medium**) to `cuda:1`. *Outcome:* High accuracy (>0.75 confidence) with zero latency penalty.
- [x] **Action 2.2: Logic-Based Classification Layer (v1).**
    - *Status:* **Implemented but Flawed.** Simple area thresholds ($Area < 35k$) cause flickering when vehicles move away (Perspective Distortion).
- [x] **Action 2.3: Tracking.**
    - *Implementation:* **ByteTrack** integrated.
    - *Refinement:* Added **Horizon Filter** (y < 0.35) to remove distant/noisy detections.
- [x] **Action 2.4: Basic Counting.**
    - *Status:* **Active but Flawed.** Vector-based line crossing working, but counts are polluted by classification flickering (Truck $\leftrightarrow$ MCV) and double-counting (tracker ID switches).

#### 🔄 Active Remediation Actions (The Logic Overhaul - Sequenced for Execution)
- [x] **Action 2.5: Hardware-Accelerated Video Export (Priority 0 - Infrastructure).**
    - *Problem:* `cv2.VideoWriter` generates massive, uncompressed files (GBs instead of MBs) that clog the drive during testing.
    - *Strategy:* Replaced OpenCV writer with a piped **FFmpeg Subprocess** using `hevc_nvenc` to offload compression to the GPU media engine for extreme size reduction.
- [ ] **Action 2.6: Perspective Correction Logic (Priority 1 - Data Integrity).**
    - *Problem:* Distant Heavy Trucks are mathematically misclassified as MCV because their pixel area drops below the static threshold.
    - *Strategy:* **Reference Box Calibration.** Define `NEAR_SCALE` and `FAR_SCALE`. Implement `normalize_area()` to convert raw pixels to depth-invariant "Virtual Meters" before applying logic thresholds.
- [ ] **Action 2.7: Double-Counting Mitigation (Priority 2 - Data Integrity).**
    - *Problem:* Same vehicles (e.g., Jeeps) counted twice due to ID switching near the line or bounding box wobble.
    - *Strategy:* Increase ByteTrack `track_buffer` (e.g., to 60 frames) to survive occlusion. Implement a `counted_ids` Set to permanently lock an ID from triggering the counter more than once.
- [ ] **Action 2.8: The "Attribute" Classifier (Priority 3 - Enhancement).**
    - *Problem:* Heuristic rules (Yellow Color for Taxi) are inaccurate due to lighting.
    - *Strategy:* **Secondary Classification Model.** Train a tiny, fast classifier (ResNet18) to distinguish `Taxi` vs `Private` vs `Jeep` based on crops.
- [ ] **Action 2.9: Auto-Calibration / The "Smart" Line (Priority 4 - QoL).**
    - *Problem:* Manual line coordinates are tedious.
    - *Strategy:* **Flow-Based Auto-Setup**. Run a 10-second "Calibration Pass" to detect average vehicle trajectory and automatically generate the optimal Counting Line.

### Phase 3: Hardware Orchestration (Dual-GPU)
**Status:** ⏳ PENDING
- [ ] Implement NVIDIA DALI for GPU-accelerated video decoding.
- [ ] Implement Python Multiprocessing Worker Strategy (1 Worker per GPU).

### Phase 4: Data Export & Validation
**Status:** ⏳ PENDING
- [ ] Generate per-class CSV/JSON summary reports.

---

## 🏗️ Architectural Decisions (Log)
| Date | Decision | Rationale |
| :--- | :--- | :--- |
| 2026-01-26 | **Docker-First Environment** | Avoids "dependency hell" on host OS; ensures GPU passthrough stability. |
| 2026-01-26 | **Dual-GPU Worker Strategy** | Maximizes 24GB total VRAM via parallel file processing. |
| 2026-01-28 | **Container Downgrade** | Downgraded to Ubuntu 22.04 / CUDA 12.4 to ensure stable OpenCV compilation. |
| 2026-01-28 | **Decoding Pivot (DALI)** | Switched to NVIDIA DALI for maintenance-free GPU decoding, abandoning `NVCUVID`. |
| 2026-01-30 | **Config-Driven Taxonomy** | Decoupled Class Mapping from Python code. `vehicle_taxonomy.json` is now the Single Source of Truth. |
| 2026-01-30 | **Manifest Oversampling** | Solved <1% class imbalance (Mini Bus) by repeating file paths in the training list 20x. |
| 2026-01-31 | **The "Medium" Pivot (Training)** | Switched Training to YOLO11-Medium to resolve 5-day bottleneck. |
| 2026-01-31 | **Ensemble Inference** | Adopted "Sidecar" strategy (Vehicle Model + Pedestrian Model). |
| 2026-01-31 | **ByteTrack Adoption** | Switched from BoT-SORT to ByteTrack due to high detection confidence. |
| 2026-02-04 | **Universal Vehicle Suppression** | Logic Rule: If a "Person" overlaps >30% with *any* vehicle box, suppress them (Driver/Passenger/False Positive). |
| 2026-02-04 | **Dual-GPU Isolation (Inference)** | Assigned Vehicle Model to `GPU 0` and Pedestrian Model to `GPU 1`. Allowed upgrading Pedestrian model to **Medium** for high accuracy without checking VRAM limits. |
| 2026-02-05 | **Perspective Correction** | Decided to abandon static "Area Thresholds" for MCV/Truck logic. Moving to **Depth-Normalized Area** to handle perspective distortion. |
| 2026-02-26 | **Double-Counting Stateful Locks** | Transitioning from basic coordinate tracking to a `counted_ids` Set memory structure to prevent tracker jitter from artificially inflating volume metrics. |
| 2026-02-26 | **FFmpeg Export Pipe** | Moved away from OpenCV `VideoWriter` to hardware-accelerated FFmpeg (`hevc_nvenc`) to solve file size bloat. |

---

## ⚠️ Lessons Learned (The "Logic" Pivot)
- **Rule-Based Failure:** Simple geometry (Area, Ratio) fails when depth changes. A "Small Truck" and a "Distant Big Truck" look identical in pixel count.
    - *Lesson:* We must normalize size based on Y-position (Depth) before classifying.
- **State Persistence Needs Buffer:** Relying purely on track IDs for counting fails if the tracker drops the object.
    - *Lesson:* We must increase the tracker's memory buffer to bridge gaps caused by occlusion, and use a strict state lock (`counted_ids`) so an object is mathematically immune to double-counting.
- **OpenCV VideoWriter Bloat:** Default OpenCV video encoding writes massive, uncompressed files.
    - *Lesson:* Always use a subprocess pipe to an external hardware encoder (FFmpeg NVENC) for AI video output.