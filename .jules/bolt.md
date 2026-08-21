
## 2024-05-18 - Memoizing heavy calculations on rapid re-renders
**Learning:** Polling dashboards combined with frequent telemetry events can cause a lot of rapid re-renders. Sorting a list (O(n log n)) inside a component that renders frequently on an interval can add noticeable CPU load on low-end devices and UI jank.
**Action:** Use `useMemo` for derived lists and sorts that only change when the underlying collection's referential identity changes, especially in dashboards with frequent intervals.
## 2024-11-20 - [Performance] **Learning:** Repeated polling/web sockets triggering UI re-renders **Action:** In NextJS frontend, ensure components handling arrays like `messages` and derived sub-lists (`crossings`) utilize `useMemo` specifically targeting the sliced views or maps to avoid unnecessary O(N) evaluations in tight polling loops.
## 2024-06-25 - Caching base date in telemetry message loops
**Learning:** In telemetry parsing (`crossingsFromTelemetry`), avoiding redundant regex and Date instantiation inside high-frequency message loops prevents significant performance degradation during live monitoring.
**Action:** When iterating over real-time or bulk events in UI formatters, lift any static parsings (e.g. baseline timestamps from job metadata) out of the loop and reuse a pre-calculated reference.
- Replaced recursive `rglob` with targeted `glob` in `resolve_preview_path`
