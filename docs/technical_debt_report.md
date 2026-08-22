# Technical Debt & Cleanup Report

## 🔴 High Confidence (Safe to Delete)
* **Dead Code:** `find_best_frame_offset` in `src/viana/stages/prescan.py` is unused and not referenced in the application (except for a test which can also be removed if it tests this dead code).
* **Misplaced Test Utils:** `RecordingRenderer` in `src/viana/stages/render.py` is a test helper that leaked into production code. It should be moved to `tests/viana/test_process.py` where it is actually used.

## 🟡 Medium Confidence (Needs Human Review)
* **Unused Worker Pool Helper:** `stub_awaiting_review` in `src/orchestrator/workers/pool.py` is explicitly commented as a "Step 2 test helper" and is only used in tests. Consider moving it to test files if it's not a public API meant to be used.
* **Test Utility Methods on Pool:** `wait_job`, `wait_for_status`, and `occupied_gpus` on `src/orchestrator/workers/pool.py` appear to be used exclusively in pytest files.
* **False Positives (Vulture):**
    * Pydantic `@field_validator` and `@model_validator` methods (e.g. `validate_gpu_device`, `validate_project_id`, `non_negative_pixels`, `frame_within_total`, etc.).
    * FastAPI route handler functions (e.g. `get_source_video`, `post_jobs_intake`, `post_job`, etc.).
    * `write_rows` in `src/viana/io/events.py` is a valid batch helper, though it is currently only invoked in tests.
    * `next_boundary_delta_ms` in `src/viana/stages/time_map.py` is a utility only invoked in tests.

## 📦 Unused Dependencies
* **Missing Definitions:** `cv2`, `numpy`, `easyocr`, `ultralytics`, `trackers`, `supervision` are imported in the CV engine but missing from `pyproject.toml` or `requirements.txt`. They might be implicitly installed or missing.
* (Note: `deptry` flagged standard tools like `pytest`, `uvicorn`, `mypy` which are correctly placed in the dev block or used for standard runtime, but appear unused in direct imports).
