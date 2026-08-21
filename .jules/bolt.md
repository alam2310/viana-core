
## 2024-05-18 - Memoizing heavy calculations on rapid re-renders
**Learning:** Polling dashboards combined with frequent telemetry events can cause a lot of rapid re-renders. Sorting a list (O(n log n)) inside a component that renders frequently on an interval can add noticeable CPU load on low-end devices and UI jank.
**Action:** Use `useMemo` for derived lists and sorts that only change when the underlying collection's referential identity changes, especially in dashboards with frequent intervals.
## 2024-11-20 - [Performance] **Learning:** Repeated polling/web sockets triggering UI re-renders **Action:** In NextJS frontend, ensure components handling arrays like `messages` and derived sub-lists (`crossings`) utilize `useMemo` specifically targeting the sliced views or maps to avoid unnecessary O(N) evaluations in tight polling loops.
## 2024-06-25 - Caching base date in telemetry message loops
**Learning:** In telemetry parsing (`crossingsFromTelemetry`), avoiding redundant regex and Date instantiation inside high-frequency message loops prevents significant performance degradation during live monitoring.
**Action:** When iterating over real-time or bulk events in UI formatters, lift any static parsings (e.g. baseline timestamps from job metadata) out of the loop and reuse a pre-calculated reference.

### Threading Optimization & Concurrency Safety
* **O(N) Operations in Lock Boundaries:** When dealing with multi-threaded components (like `WorkerPool` in `src/orchestrator/workers/pool.py`), placing $O(N)$ scanning operations inside tight `while True:` loop lock boundaries leads to catastrophic CPU utilization bottlenecks as the active queue size increases.
* **Class-Level State Tracking:** Rather than scanning large state structures (like looping through all jobs to identify occupied GPUs or running prescans) upon every loop iteration, explicitly manage application state changes (in `job.status`) via a single setter method (e.g., `_set_job_status(job, new_status)`). This allows for atomic updates to caching properties (like `self._occupied_gpus` and `self._running_prescans`), driving $O(N)$ operations down to $O(1)$.
* **Stale Cached State Hazards:** Never read global state from a localized variable cache outside of a thread lock boundary while yielding or blocking. This explicitly leads to race condition states when multiple threads manipulate the process pool simultaneously, particularly breaking GPU allocation isolation constraints. Always access up-to-date state synchronously under the thread lock right when it is needed.
- Replaced recursive `rglob` with targeted `glob` in `resolve_preview_path`
