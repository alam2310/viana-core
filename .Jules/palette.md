## 2026-08-20 - Missing Keyboard Focus States on Icon Buttons
**Learning:** Custom interactive components like `RoundIconButton` must explicitly define `focus-visible` utility classes to support keyboard accessibility, as they don't inherit them automatically from default HTML elements or other UI components like `Button`.
**Action:** Ensure all interactive elements, especially custom ones, include explicit `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border` classes to match the design system's focus indication pattern.
