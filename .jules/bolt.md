## 2025-02-12 - NMS O(N^2) CPU Bounding
**Learning:** In purely Python, function call overhead inside nested loops processing thousands of items can dominate runtime. Calling modular helper functions like `iou()` inside `nms_class_agnostic` made the algorithm 6x slower due to the overhead of parameter passing, function scope resolution, and missing early-bailouts.
**Action:** Always inline simple math and check bounding boxes with simple conditionals (e.g. AABB intersection test) before calculating precise areas when inside tight loops (like O(N^2) NMS).
