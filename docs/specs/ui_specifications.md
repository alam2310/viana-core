# UI Specifications (Next.js 15)

## 1. Container Lifecycle Management
*   **Startup Verification:** Before submitting jobs from the queue, the UI must check if the backend container is running.
*   **Auto-Start:** If the container is not running (or no job is in a running state and the container was spun down), the Next.js API must start it using the parameters defined in the local `orchestrator_config.yaml` file (or `.env` equivalent).
*   **Context Preservation:** The jobs submitted from the UI must use the exact context, volume mounts, and network ports exposed by this locally managed container.

## 2. Global Queue & State Management
*   **Asynchronous Batching:** The UI must support queueing up to 50+ videos. Adding a job to the queue is non-blocking. 
*   **State Persistence:** The `PENDING`, `PROCESSING`, and `COMPLETED` queues must be synced to local storage to survive browser refreshes or accidental closure.
*   **Dashboard Layout:** A sidebar for multi-job status monitoring (progress bars) and a main viewport for the active job (live video canvas and dynamic telemetry console).

## 3. Pre-Screening Workflow
1.  **File Input:** User inputs a local video file path.
2.  **API Pre-Scan:** UI calls `/utils/prescan` on the container context. The container samples the first 5-10 seconds, runs OSD OCR, and returns extracted metadata.
3.  **Review Modal:** 
    *   Displays extracted Time/Location for user validation.
    *   Includes a **Video Frame Scrubber** to skip dark/blocked opening frames.
4.  **Task Calibration (HTML5 Canvas):**
    *   *ViAna:* User draws 2 vector lines (`Horizon`, `Counting`).
    *   *ViAnaNP:* No drawing required; metadata verification only.
    *   *ViAnaJunction:* User draws a polygon and defines $N$ named edge gates.
    *   *Feature:* "Apply to all pending" checkbox to duplicate vector coordinates to subsequent videos in the batch.

## 4. UI Component Toolkit
*   Use Tailwind v4 for all styling and responsive grid layouts.
*   Utilize Shadcn/UI for modals, forms, toast notifications, and dropdowns.
*   Live telemetry tables should auto-scroll and render highly optimized lists to prevent DOM lag when receiving high-frequency WebSocket updates.