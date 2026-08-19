# UX discovery (Step 1 — living document)

**Status:** 🔄 In progress (Phase 1.1)  
**Owner:** Step 1 UX chat  
**Output:** feeds [`REDESIGN.md`](REDESIGN.md) when discovery is complete

> The UX agent maintains this file during Step 1. Record answers, decisions, and open questions here before finalizing the redesign spec.

---

## 1. Product context (agent fills from docs)

| Topic | Summary |
|-------|---------|
| **Platform** | Next.js UI on host orchestrates a GPU Docker container (FastAPI + `viana` CV engine). Local video paths only — no upload. |
| **v0.1 scope** | `ViAna_Moving` only — directional moving vehicle count via horizon + counting lines. |
| **Future tasks** | `ViAnaNP` (parked extraction, metadata-only calibration); `ViAnaJunction` (polygon + O/D gates). Design for extensibility; not shipped in v0.1. |
| **Prescan today** | Engine samples frame (configurable offset), runs OSD OCR (time/date/location), proposes horizon + counting lines (from profiles or heuristics), writes preview JPEG. API: `POST /utils/prescan`. |
| **Current UI** | Single modal: run prescan → canvas + editable OCR fields + frame scrubber → "Save calibration" (blocks on geometry validation). "Apply to pending" + optional profile save. |
| **User must confirm** | Step 1 goal: explicit **propose → confirm/edit** for time, date, location, horizon line, counting line before job submit. |
| **15-min report** | Wall-clock metadata from prescan → `time_map.json` → aggregate → `{stem}_15min.csv`. Empty CSV is a known verification target (Step 4). |
| **Hardware** | Dual RTX 3060; backend assigns `gpu_device`. Batch sizes up to 50+ videos per project (spec). |

---

## 2. Stakeholder Q&A

_Agent: ask the user questions in chat. Paste questions and answers below._

### Session log

| # | Date | Question | Answer / decision |
|---|------|----------|-------------------|
| — | 2026-08-19 | _(Batch 1 sent — awaiting answers)_ | — |

### Open questions

- [ ] Who are the primary operators and typical batch size?
- [ ] Are time/date/location mandatory before submit, or can jobs run with blanks?
- [ ] Low OCR confidence UX — warn, block, or allow with acknowledgment?
- [ ] Date/time display and input format (24h? DD-MM-YYYY? free text?)
- [ ] Trust level for auto-proposed lines vs always manual review?
- [ ] "Apply to all pending" — lines only, or metadata too?
- [ ] Required job-card fields on dashboard (queue + completed)?
- [ ] Completed job: aggregate trigger UX; empty `_15min.csv` messaging?
- [ ] Modal vs wizard for prescan; information density preferences?
- [ ] NP/Junction prescan expectations for task matrix (even if future)?

---

## 3. Task-type prescan matrix

| Task | Prescan proposes | User confirms / edits | Calibration UI | v0.1 ship? |
|------|------------------|----------------------|----------------|------------|
| **ViAna_Moving** | OCR: time, date, location; horizon + counting lines; preview frame | All five fields editable; submit blocked until geometry valid | 2-line canvas on preview JPEG | ✅ Yes |
| **ViAnaNP** | OCR metadata (time, date, location); no line geometry | Metadata verification only | No canvas (spec §3) | ❌ Future |
| **ViAnaJunction** | OCR metadata; proposed polygon + N named edge gates (TBD engine) | Polygon + gate labels/positions | Polygon + gate canvas (spec §3) | ❌ Future |

**Extensibility note:** Prescan request/response may need `task_type` so engine returns task-appropriate proposals (Step 2 if confirmed).

---

## 4. Screen inventory (big picture)

| Screen | Purpose | Notes |
|--------|---------|-------|
| **Settings / first launch** | Container health, start backend, API health check | Flow 1 in `USER_FLOWS.md` |
| **Dashboard / queue** | `project_id`, pending video paths, job list from `GET /jobs` | localStorage for prefs + pending paths only |
| **Prescan modal** | Per-video propose → confirm/edit OCR + lines before submit | Core Step 1 redesign surface |
| **Job monitor / telemetry** | Progress bars, WS events, optional detail telemetry | Virtualized table for high-frequency events |
| **Paused / failed job** | Resume vs start-fresh | Checkpoint-aware cards |
| **Completed job / artifacts** | Links to events CSV, 15-min CSV, processed MP4; aggregate trigger | Empty 15-min messaging TBD |
| **Container / settings** | Output parent dir, telemetry prefs, task type (v0.1: Moving only) | Host-side docker via Next API routes |

---

## 5. Backend / prescan gaps (for Step 2)

_If current prescan API or engine cannot support the UX, list here → copy to `docs/steps/STEP_2_BACKEND_ALIGNMENT.md` § Work items._

| ID | Gap | Surface | Step 2? |
|----|-----|---------|-------|
| G1 | `PrescanResponse.ocr` and `JobMetadata` are the same shape — no persisted distinction between **proposed** vs **user-confirmed** values (UI overwrites on load) | Contract + UI state | TBD — depends on whether we need audit trail or per-field confirm UX |
| G2 | No `task_type` on prescan request — engine always proposes Moving lines | Contract, `run_prescan()`, API route | TBD — required for NP/Junction extensibility |
| G3 | OCR `confidence` is a single scalar — no per-field confidence for time/date/location | Engine OCR, contract | TBD — depends on low-confidence UX |
| G4 | No server-side validation of date/time format before job submit | API / job validation | TBD — depends on mandatory metadata rules |
| G5 | Junction/NP prescan proposals not implemented in engine | `prescan.py`, future stages | Future — document in matrix only for v0.1 |

---

## 6. Discovery sign-off

- [ ] User reviewed task-type matrix for `ViAna_Moving`
- [ ] Open questions resolved or explicitly deferred
- [ ] Screen inventory complete
- [ ] Ready to write `REDESIGN.md`

**Signed off:** _date / note_
