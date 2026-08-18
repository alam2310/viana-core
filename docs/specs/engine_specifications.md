# Analytics Engine Specifications (Python Backend)

## 1. Container & Execution Environment
*   **Packaging:** The entire FastAPI application and computer vision dependencies must be packaged in a Docker/Podman image supporting NVIDIA CUDA (`nvidia-container-toolkit`).
*   **Volume Mounts:** The container relies on the UI Orchestrator to correctly map the local system drives (where the videos live) to the container's internal filesystem via the orchestrator's config settings.
*   **CLI Mode:** Every underlying engine must remain executable via command line *inside* the container for debugging:
    *   `python viana_np.py --run --source /data/vid.mp4 --start-time "09:00:00"`

## 2. Execution & Parallelization
*   **Worker Pool:** The FastAPI application maintains an asynchronous job queue. The concurrency limit is strictly 2 workers.
*   **GPU Isolation:** Worker 1 binds entirely to `cuda:0`. Worker 2 binds entirely to `cuda:1`. Dual-GPU orchestration is handled at the worker spawn level.

## 3. Shared Computer Vision Assets
*   **YOLO11-Medium:** Trained at 1088p resolution. 
*   **Output Encoding:** Output preview video rendering must use `hevc_nvenc` via FFmpeg wrappers to prevent CPU/CUDA blocking. 

## 4. Engine-Specific Logic Notes
*   **ViAnaNP (Parked):**
    *   Uses egomotion compensation targeting bounding boxes moving towards frame edges (scale expansion).
    *   Applies a confidence-weighted majority vote across all tracked OCR frames for a specific `Track_ID`.
    *   Applies Levenshtein distance ($\le 1$) fuzzy string matching to deduplicate noisy OCR reads within a 1-hour temporal block.
*   **ViAnaJunction:**
    *   Utilizes polygon-intersection tests (e.g., Shapely) to detect when a track centroid crosses a Gate line segment.
    *   Lifecycle tracking per ID: `Status: Entered -> Origin Logged -> Status: Exited -> Destination Logged`.