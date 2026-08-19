# Step 2 — Backend alignment (conditional)

| Field | Value |
|-------|-------|
| **Status** | ⏸ Skipped until Step 1 identifies gaps — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | **New** — Backend (Contract and/or Engine/API) |
| **Blocks** | Step 3 |
| **Blocked by** | Step 1 § Work items |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 2.

---

## Objective

Close gaps between Step 1 UX and current prescan/backend **before** UI implementation.

Step 2 runs only when Step 1 lists one or more work items. It may include:

| Track | Examples |
|-------|----------|
| **Contract** | New `PrescanResponse` fields, per-task prescan request shape |
| **Engine** | `run_prescan()` proposes different data per task; OCR/format changes |
| **API** | `POST /utils/prescan` behavior, response mapping, fixtures |

**Skip Step 2** when Step 1 UX fits existing contracts and prescan behavior.

---

## Work items (from Step 1)

_Fill from `docs/ui/DISCOVERY.md` §5 and `STEP_1` exit review._

| ID | Item | Owner | Status |
|----|------|-------|--------|
| — | _none yet_ | — | — |

---

## Contract changes

Follow `docs/governance/CONTRACT_SYNC.md`:

`schemas/` → `typescript/` → `fixtures/` → `job.py` (if needed) → `api_contracts.md` → `openapi.yaml`

### Proposals log

| ID | Field / endpoint | Schema file | Rationale | Status |
|----|------------------|-------------|-----------|--------|
| — | _none yet_ | — | — | — |

---

## Prescan implementation changes

Typical surfaces when UX requires new proposal behavior:

| Surface | Path |
|---------|------|
| Engine prescan | `src/viana/stages/prescan.py`, `src/viana/cli.py` |
| API route | `src/orchestrator/routes/` (prescan utils) |
| Tests | `tests/viana/test_prescan.py`, `tests/orchestrator/` |
| UI fixtures | `packages/contracts/fixtures/prescan_response.json` |

---

## Chat split (when Step 2 runs)

| Work | Chat |
|------|------|
| Schema + fixtures only | **Contract** |
| Engine prescan logic | **Engine** (after or with contract) |
| HTTP route / mapping | **API** |
| All three in one small gap | Single **Backend** chat OK |

Use **new** chats — not Phase 1–9 build threads.

---

## Exit criteria

- [ ] All work items ✅ or deferred with user approval
- [ ] Tests pass for touched prescan/contract paths
- [ ] `TRACKER.md` Step 2 complete or skipped
- [ ] Step 3 unblocked

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Renamed from contract-only; includes prescan engine/API |
