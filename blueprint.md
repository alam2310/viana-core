# 🗺️ Project Blueprint: Indian Traffic Video Analytics (ITVA)
**Objective:** End-to-end offline pipeline for high-accuracy vehicle classification and counting.
**Hardware:** i7-12700F | 32GB DDR5 | 2x RTX 3060 (12GB) | Host: Ubuntu 24.04 | Container: Ubuntu 22.04
**Target Classes:** 2-Wheelers, Cars, Auto Rickshaws, Trucks, Medium Vehicles, Light Vehicles, Heavy Vehicles.

---

## 🚦 Project Master Status
- **Current Phase:** Phase 2: High-Accuracy Offline Engine (Inference Logic)
- **Overall Progress:** 60% (Model Released v1.0)
- **Last Updated:** 2026-01-31

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

#### ⚠️ History of Failed Attempts (Lessons Learned)
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
    - [x] **Outcome:** Achieved **mAP@50 > 0.93** and **Recall > 0.88** by Epoch 9.
    - [x] **Validation:** Confirmed "Mini Bus" class is generalizing (Precision ~0.92, Recall ~0.98). Model released as `itva_medium_1088p.pt`.
	- [x] Completion: Stopped training at Epoch 9 due to sufficient convergence (diminishing returns).

### Phase 2: High-Accuracy Offline Engine
**Status:** 🚀 IN PROGRESS
- [ ] **Action 2.1: The "Ensemble" Inference Engine.**
    - *Strategy:* Run two models per frame to bridge the "missing class" gap.
    - *Model A:* Custom `best.pt` (Medium) for Vehicles (Auto, Tempo, Truck).
    - *Model B:* Standard `yolo11n.pt` (Nano) for **Pedestrians** (Class 0).
- [ ] **Action 2.2: Logic-Based Classification Layer.**
    - *Problem:* UVH-26 lacks explicit labels for MCV, Trailers, and Taxis.
    - *Solution:* Implement Post-Processing Heuristics:
        - **MCV/Heavy Split:** Filter `Truck` class by Bounding Box Area ($Area < T_{mcv} \rightarrow MCV$).
        - **Trailer Split:** Filter `Truck` class by Aspect Ratio ($W/H > 2.5 \rightarrow Trailer$).
        - **Taxi Split:** Filter `Car` by Color (Yellow Region Analysis).
- [ ] **Action 2.3: High-Speed Tracking.**
    - *Decision:* Use **ByteTrack** instead of BoT-SORT.
    - *Reason:* High detection confidence (mAP50-95 ~0.85) makes Re-ID redundant. ByteTrack saves VRAM and is faster.
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
| 2026-01-26 | **Docker-First Environment** | Avoids "dependency hell" on host OS; ensures GPU passthrough stability. |
| 2026-01-26 | **Dual-GPU Worker Strategy** | Maximizes 24GB total VRAM via parallel file processing. |
| 2026-01-28 | **Container Downgrade** | Downgraded to Ubuntu 22.04 / CUDA 12.4 to ensure stable OpenCV compilation. |
| 2026-01-28 | **Decoding Pivot (DALI)** | Switched to NVIDIA DALI for maintenance-free GPU decoding, abandoning `NVCUVID`. |
| 2026-01-30 | **Config-Driven Taxonomy** | Decoupled Class Mapping from Python code. `vehicle_taxonomy.json` is now the Single Source of Truth. |
| 2026-01-30 | **Manifest Oversampling** | Solved <1% class imbalance (Mini Bus) by repeating file paths in the training list 20x. |
| 2026-01-31 | **The "Medium" Pivot** | Switched from Large to Medium Model. Drastically improved training speed while maintaining >0.92 mAP via 1088p resolution. |
| 2026-01-31 | **Ensemble Inference** | Adopted "Sidecar" strategy: Running a parallel YOLO-Nano model specifically for Pedestrians, avoiding the need to retrain for common classes. |
| 2026-01-31 | **Native Inference (No SAHI)** | Dropped SAHI for Phase 2. Native 1088p Recall (0.88+) is sufficient; SAHI latency is unjustified. |
| 2026-01-31 | **ByteTrack Adoption** | Switched from BoT-SORT to ByteTrack due to high detection confidence (mAP50-95 > 0.85), saving compute resources. |
| 2026-01-31 | **Early Stopping (Epoch 9)** | Model reached "Production Grade" metrics (mAP > 0.93) rapidly. Halted training to prevent overfitting and move to Inference logic. |

---

## ⚠️ Known Blockers / Risks
- **Overfitting Monitor:** Rapid convergence (Epoch 7) suggests potential memorization of oversampled classes. 
    - *Status:* Monitor Class-wise Precision. Current metrics show healthy generalization (Val Loss decreasing).
- **Logic Fragility:** "Heuristic" splitting (e.g., Truck vs MCV by size) is sensitive to camera distance/calibration.
    - *Mitigation:* Will require a "Calibration Factor" in the config for each video source.