
## 2024-05-18 - Memoizing heavy calculations on rapid re-renders
**Learning:** Polling dashboards combined with frequent telemetry events can cause a lot of rapid re-renders. Sorting a list (O(n log n)) inside a component that renders frequently on an interval can add noticeable CPU load on low-end devices and UI jank.
**Action:** Use `useMemo` for derived lists and sorts that only change when the underlying collection's referential identity changes, especially in dashboards with frequent intervals.
