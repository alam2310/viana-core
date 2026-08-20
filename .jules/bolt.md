## 2024-06-25 - Caching base date in telemetry message loops
**Learning:** In telemetry parsing (`crossingsFromTelemetry`), avoiding redundant regex and Date instantiation inside high-frequency message loops prevents significant performance degradation during live monitoring.
**Action:** When iterating over real-time or bulk events in UI formatters, lift any static parsings (e.g. baseline timestamps from job metadata) out of the loop and reuse a pre-calculated reference.
