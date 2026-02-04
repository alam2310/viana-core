# 🗺️ Project Blueprint: Indian Traffic Video Analytics (ITVA)
**Objective:** End-to-end offline pipeline for high-accuracy vehicle classification and counting.
**Hardware:** i7-12700F | 32GB DDR5 | 2x RTX 3060 (12GB) | Host: Ubuntu 24.04 | Container: Ubuntu 22.04
**Target Classes:** 2-Wheelers, Cars, Auto Rickshaws, Trucks, Medium Vehicles, Light Vehicles, Heavy Vehicles.

---

## 🚦 Project Master Status
- **Current Phase:** Phase 2: High-Accuracy Offline Engine (Inference Logic)
- **Overall Progress:** 60% (Inference Engine Stabilized)
- **Last Updated:** 2026-02-04

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

#### ⚠️ History of Failed Attempts (Phase 0)
- [❌] **FAILED:** Build "Bleeding Edge" Image (Ubuntu 24.04 + CUDA 13.1). 
- [❌] **FAILED:** Compile OpenCV 4.10 with `NVCUVID=ON` inside Docker.

### Phase 1: Model & Dataset Strategy
**Status:** ✅ COMPLETE
- [x] **Action 1.1: Audit UVH-26 Dataset.** (Identified Mini Bus imbalance).
- [x] **Action 1.2: Design Taxonomy & Logic.** (Config-Driven `vehicle_taxonomy.json`).
- [x] **Action 1.3: Engineer Balanced Dataset.** (Manifest Oversampling 20x).
- [x] **Action 1.4: Train Model (The "Brain").**
    - [x] **Pivot:** Switched to **YOLO11-Medium** @ 1088p.
    - [x] **Outcome:** Achieved **mAP@50 > 0.92** and **Recall > 0.88** by Epoch 7.
    - [x] **Validation:** Confirmed "Mini Bus" generalization.

### Phase 2: High-Accuracy Offline Engine
**Status:** 🚀 IN PROGRESS
- [x] **Action 2.1: The "Ensemble" Inference Engine.**
    - *Goal:* Run two models simultaneously to cover Vehicles + Pedestrians.
    - **Iteration Log (The Accuracy Climb):**
        - [x] **Attempt 1 (Nano Sidecar):** Used `yolo11n.pt` for pedestrians. 
            - *Result:* High speed, but false positives on Motorbike Riders and Car parts. Low confidence (0.35).
        - [x] **Attempt 2 (Rider Suppression):** Implemented IoA Logic (`Intersection / Person_Area > 0.3`) to suppress people overlapping with 2-Wheelers.
            - *Result:* Fixed "Rider" issue, but "Car Side" false positives remained.
        - [x] **Attempt 3 (Small + Universal Suppression):** Upgraded to `yolo11s.pt` and checked overlap against *ALL* vehicles.
            - *Result:* Better, but confidence still borderline (0.40 - 0.65).
        - [x] **Attempt 4 (Dual-GPU Isolation):** **Final Strategy.**
            - *Implementation:* Pushed Model A (Vehicles) to `cuda:0` and Model B (Pedestrians, **YOLO11-Medium**) to `cuda:1`.
            - *Outcome:* High accuracy (>0.75 confidence) with zero latency penalty on the main vehicle stream.
- [ ] **Action 2.2: Logic-Based Classification Layer.**
    - *Problem:* UVH-26 lacks explicit labels for MCV, Trailers, and Taxis.
    - *Solution:* Implement Post-Processing Heuristics:
        - **MCV/Heavy Split:** Filter `Truck` class by Bounding Box Area ($Area < T_{mcv} \rightarrow MCV$).
        - **Trailer Split:** Filter `Truck` class by Aspect Ratio ($W/H > 2.5 \rightarrow Trailer$).
        - **Taxi Split:** Filter `Car` by Color (Yellow Region Analysis).
- [X] **Action 2.3: High-Speed Tracking.**
    - *Decision:* Use **ByteTrack** instead of BoT-SORT.
- [ ] **Action 2.4: Counting Logic.**
    - *Implementation:* Vector-based Line Crossing (In/Out counts).

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
| 2026-01-28 | **Decoding Pivot (DALI)** | Switched to NVIDIA DALI for maintenance-free GPU decoding. |
| 2026-01-30 | **Config-Driven Taxonomy** | Decoupled Class Mapping from Python code. |
| 2026-01-31 | **The "Medium" Pivot (Training)** | Switched Training to YOLO11-Medium to resolve 5-day bottleneck. |
| 2026-01-31 | **Ensemble Inference** | Adopted "Sidecar" strategy (Vehicle Model + Pedestrian Model). |
| 2026-01-31 | **ByteTrack Adoption** | Switched from BoT-SORT to ByteTrack due to high detection confidence. |
| 2026-02-04 | **Universal Vehicle Suppression** | Logic Rule: If a "Person" overlaps >30% with *any* vehicle box, suppress them (Driver/Passenger/False Positive). |
| 2026-02-04 | **Dual-GPU Isolation (Inference)** | Assigned Vehicle Model to `GPU 0` and Pedestrian Model to `GPU 1`. Allowed upgrading Pedestrian model to **Medium** for high accuracy without checking VRAM limits. |

---

## ⚠️ Known Blockers / Risks
- **Logic Fragility:** "Heuristic" splitting (e.g., Truck vs MCV by size) is sensitive to camera distance/calibration.
    - *Mitigation:* Will require a "Calibration Factor" in the config for each video source.
- **Distance Decay:** Pedestrians/Bikes far away (>50m) are missed or misclassified.
    - *Mitigation:* Implemented **Horizon Line Filter** (Action 2.3) to ignore unreliable detections in the upper frame background.