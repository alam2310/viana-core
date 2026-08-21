## 2024-03-24 - Focus States Added
**Learning:** Found that keyboard navigation was hindered due to missing focus states on `RoundIconButton` and clickable file lists. Relying on default focus states isn't enough when CSS reset overrides them.
**Action:** Always ensure `focus-visible` classes are added to custom interactive elements.
## 2026-08-21 - Focus Visible Requirements for Inputs, Selects, and Pagination
**Learning:** The application extensively utilizes 'focus-visible' utility classes to meet keyboard accessibility standards. In addition to primary buttons, pagination controls, text inputs, and select dropdowns must consistently apply 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border' to ensure proper visual feedback for keyboard users.
**Action:** Apply these standard focus-visible classes to all newly created or modified interactive elements, not just icon buttons.
