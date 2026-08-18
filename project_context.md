# Project Context: Unified Video Analytics Platform

## 1. System Architecture overview
This project consists of a decoupled architecture with a web-based UI Orchestrator managing multiple headless Computer Vision (CV) backend engines. 
**Crucial Architectural Note:** The backend CV engines are packaged inside a Docker/Podman container. The Next.js UI acts not just as a frontend, but as a local system orchestrator that actively manages the lifecycle of this backend container.

## 2. Core Technologies & Hardware
*   **Target Hardware:** Intel Core i7-12700F, 32GB RAM, Dual NVIDIA RTX 3060 (12GB) GPUs.
*   **Containerization:** NVIDIA Container Toolkit (Docker/Podman) for passing GPU access (`cuda:0`, `cuda:1`) into the backend environment.
*   **Frontend UI & Orchestrator:** Next.js 15, Tailwind v4, Shadcn/UI. (Next.js API routes execute local host commands to manage the container).
*   **Backend API (Inside Container):** FastAPI (Python), WebSockets, Asyncio.
*   **CV Engines:** Ultralytics (YOLO11), ByteTrack/DeepSORT, EasyOCR/Tesseract.

## 3. The Analytics Engines (Multi-Tenant Worker Pool)
The containerized backend dynamically assigns jobs from a pending queue to isolated GPU workers.
1.  **ViAna (Moving Count):** Directional moving vehicle counting via Horizon/Counting lines.
2.  **ViAnaNP (Parked Extraction):** Egomotion-based parked vehicle extraction, multi-frame OCR, and 1-hour deduplication.
3.  **ViAnaJunction (Junction Directional Count):** Tracks vehicle trajectories across a central polygon and user-defined Origin/Destination gates.

## 4. Design Principles
*   **Container-First Backend:** The backend is stateless and runs fully inside a GPU-enabled container.
*   **UI-Driven Container Lifecycle:** The UI must verify container health and automatically start the container if it is down or idle before submitting jobs.
*   **Automated Output Routing:** Artifacts are written to `{output.parent_dir}/{project_id}/` (default `/data/viana-outputs/{project_id}/`). See `docs/ui/OUTPUT_PATHS.md`.
*   **Backend Job Management:** `job_id` and `gpu_device` are assigned by the FastAPI orchestrator, not the UI.
*   **Contract-Driven Development:** The frontend and backend communicate strictly via standardized JSON and WebSocket payloads (`api_contracts.md`, `packages/contracts/`).
*   **Monorepo:** Engine (`src/viana/`), orchestrator (`src/orchestrator/`), UI (`apps/web/`), shared contracts (`packages/contracts/`).
*   **AI SDLC:** See `AGENTS.md`, `docs/governance/AI_SDLC.md`, `docs/PROJECT_STATUS.md`.