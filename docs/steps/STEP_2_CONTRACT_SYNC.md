# Step 2 — Contract sync (conditional)

| Field | Value |
|-------|-------|
| **Status** | ⏸ Skipped until Step 1 proposes changes — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | New — **Contract** |
| **Blocks** | Step 3 (only if schema changes required) |
| **Blocked by** | Step 1 contract proposals |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 2.

---

## Objective

Apply schema-first contract updates when Step 1 (or Step 3 discovery) requires new or changed API/data shapes.

**Default:** Step 2 is **not needed** — `JobMetadata`, `PrescanResponse.ocr`, and aggregate already exist.

---

## When to run Step 2

| Trigger | Example |
|---------|---------|
| New job submit field | Separate `proposed_metadata` from user override |
| New prescan response field | OCR confidence breakdown |
| CSV column change | `events_15min.schema.json` |

Layout/copy-only UI changes → **skip Step 2**.

---

## Workflow

Follow `docs/governance/CONTRACT_SYNC.md` (schemas → TS → fixtures → Pydantic → docs → implement).

---

## Proposals (from Step 1)

| ID | Field / endpoint | Schema file | Rationale | Status |
|----|------------------|-------------|-----------|--------|
| — | _none yet_ | — | — | — |

---

## Exit criteria

- [ ] Proposals implemented or deferred
- [ ] `TRACKER.md` Step 2 gate complete or skipped
- [ ] Step 3 unblocked
- [ ] `AGENT_PROGRESS.md` Step 2 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step created (numbered Steps 1–5) |
