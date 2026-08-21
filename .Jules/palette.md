## 2026-08-21 - Added accessible focus ring states to custom UI components
**Learning:** It was found that custom components like 'RoundIconButton' were missing standard focus-visible rings out of the box, making keyboard navigation harder.
**Action:** Ensure any new custom interactive components get standard focus ring utility classes (e.g. `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border`).
