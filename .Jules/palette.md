## 2024-03-24 - Focus States Added
**Learning:** Found that keyboard navigation was hindered due to missing focus states on `RoundIconButton` and clickable file lists. Relying on default focus states isn't enough when CSS reset overrides them.
**Action:** Always ensure `focus-visible` classes are added to custom interactive elements.
