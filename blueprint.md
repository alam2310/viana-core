# 🗺️ Project Blueprint: Indian Traffic Video Analytics (ITVA)
**Objective:** End-to-end offline pipeline for high-accuracy vehicle classification and counting.
**Hardware:** i7-12700F | 32GB DDR5 | 2x RTX 3060 (12GB) | Host: Ubuntu 24.04 | Container: Ubuntu 22.04
**Target Classes:** 2-Wheelers, Cars, Auto Rickshaws, Trucks, Medium Vehicles, Light Vehicles, Heavy Vehicles.

---

## 🚦 Project Master Status
- **Current Phase:** Phase 1: Model & Dataset Strategy (Ready for Training)
- **Overall Progress:** 40% (Dataset Engineered & Balanced)
- **Last Updated:** 2026-01-30

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
**Status:** 🚀 IN PROGRESS
- [x] **Action 1.1: Audit UVH-26 Dataset.**
    - *Outcome:* Identified critical imbalance (Mini Bus < 1% vs MTW 47%).
    - *Artifact:* `src/utils/dataset_auditor.py`.
- [x] **Action 1.2: Design Taxonomy & Logic.**
    - *Strategy:* **Config-Driven Architecture**. Decoupled logic from code using `vehicle_taxonomy.json`.
    - *Outcome:* Normalized chaotic labels (Tempo, Tata Ace) into structured Target IDs.
- [x] **Action 1.3: Engineer Balanced Dataset.**
    - *Strategy:* **Manifest Oversampling**. Multiplied rare classes (Mini Bus 20x, LCV 5x) in the training manifest.
    - *Outcome:* `itva_phase1` dataset created. Effective balance achieved without synthetic data.
- [ ] **Action 1.4: Train Model (The "Brain").**
    - [ ] **Configuration:** YOLO11-Large (L) @ **1280px** Resolution.
    - [ ] **Hardware Strategy:** Distributed Data Parallel (DDP) on Dual RTX 3060 (`device=0,1`).
    - [ ] **Hyperparameters:**
        - `mosaic: 1.0`, `mixup: 0.2` (Aggressive augmentation to prevent overfitting 20x duplicates).
        - `box: 7.5` (High precision focus).
    - [ ] **Validation:** Verify Recall > 0.85 for "Auto Rickshaw" and "Mini Bus".

### Phase 2: High-Accuracy Offline Engine
**Status:** ⏳ PENDING
- [ ] Implement SAHI (Slicing Aided Hyper Inference) logic.
- [ ] Integrate BoT-SORT tracker with Re-ID.
- [ ] Develop "Double-Gate" counting logic.

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
| 2026-01-30 | **Manifest Oversampling** | Solved <1% class imbalance (Mini Bus) by repeating file paths in the training list 20x, avoiding synthetic data generation. |

---

## ⚠️ Known Blockers / Risks
- **Overfitting Risk:** The 20x duplication of "Mini Bus" images may cause the model to memorize specific vehicles.
    - *Mitigation:* We will use **Aggressive Mosaic/MixUp** augmentation during training (Phase 1.4) to vary the context of these duplicates.
- **Occlusion:** "Auto Rickshaws" in dense traffic are heavily occluded.
    - *Mitigation:* Training at **1280px** resolution is mandatory to resolve small features.