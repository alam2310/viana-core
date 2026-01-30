# 🗺️ Project Blueprint: Indian Traffic Video Analytics (ITVA)
**Objective:** End-to-end offline pipeline for high-accuracy vehicle classification and counting.
**Hardware:** i7-12700F | 32GB DDR5 | 2x RTX 3060 (12GB) | Host: Ubuntu 24.04 | Container: Ubuntu 22.04
**Target Classes:** 2-Wheelers, Cars, Auto Rickshaws, Trucks, Medium Vehicles, Light Vehicles, Heavy Vehicles.

---

## 🚦 Project Master Status
- **Current Phase:** Phase 1: Model & Dataset Strategy
- **Overall Progress:** 20% (Phase 0 Complete & Verified)
- **Last Updated:** 2026-01-28

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

#### ⚠️ History of Failed Attempts (Lessons Learned)
*These approaches were attempted and discarded to ensure stability.*
- [❌] **FAILED:** Build "Bleeding Edge" Image (Ubuntu 24.04 + CUDA 13.1). 
    - *Reason:* Too many breaking API changes in CUDA 13.1; incompatible with current OpenCV source.
- [❌] **FAILED:** Compile OpenCV 4.10 with `NVCUVID=ON` inside Docker.
    - *Reason:* NVIDIA deprecated headers (nvcuvid.h) in CUDA 12.x. "Headless" build environments cannot reliably satisfy linker checks for runtime drivers without fragile hacks.

### Phase 1: Model & Dataset Strategy
**Status:** 🚀 IN PROGRESS
- [ ] **Action:** Select & Audit Datasets (Targeting IISc UVH-26 or IDD).
- [ ] **Action:** Define Model Architecture (YOLOv8/v9 vs. YOLO-World).
- [ ] **Action:** Establish Baseline Validation Metrics for "Auto Rickshaws".

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
| 2026-01-28 | **Container Downgrade (Safe Landing)** | Downgraded container to **Ubuntu 22.04 / CUDA 12.4** to ensure stable OpenCV compilation without source hacking. |
| 2026-01-28 | **Decoding Pivot (DALI/FFmpeg)** | Abandoned compiling deprecated `NVCUVID` into OpenCV. Switched to **NVIDIA DALI** for production-grade, maintenance-free GPU decoding. |

---

## ⚠️ Known Blockers / Risks
- **Dataset Bias:** Indian traffic datasets often lack "Auto Rickshaw" diversity. Requires careful validation in Phase 1.
