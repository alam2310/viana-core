# Contract Sync — Cross-Track Rules (UI / Engine / API)

**Applies when Engine (Phases 1–5), UI (Phases 7–8), and API (Phase 6) run in parallel.**

The repo is the coordination layer. Chat history is not.

---

## Golden rule

> **`packages/contracts/schemas/` wins.** No track may ship code that invents, renames, or omits contract fields without updating schemas first.

---

## Who changes contracts?

| Initiator | Typical change | Must also update |
|-----------|----------------|------------------|
| **UI** needs a field for a screen | Request or PR to `packages/contracts/schemas/` | `typescript/`, fixture JSON, `docs/api_contracts.md` |
| **API** needs a response shape | Schema + fixture first | `src/viana/config/job.py` if submit payload changes |
| **Engine** needs a new artifact column | `events_*.schema.json` first | `docs/api_contracts.md`, aggregation docs |

**Any agent** may edit `packages/contracts/`, but the change must land (or be explicitly stubbed with fixture) **before** other tracks depend on it.

---

## Mandatory workflow (no exceptions)

```
1. packages/contracts/schemas/*.json     ← authoritative
2. packages/contracts/typescript/index.ts
3. packages/contracts/fixtures/*.json    ← if UI needs to mock it
4. src/viana/config/job.py               ← if job submit/response shape changes
5. docs/api_contracts.md                   ← human summary
6. openapi.yaml                           ← if HTTP surface changes
7. docs/PROJECT_STATUS.md                 ← API matrix / phase notes
8. Implement consumer (UI route, API handler, or engine)
```

---

## Track-specific constraints

### UI (`apps/web/`)

- **Must** use `packages/contracts/fixtures/` when `PROJECT_STATUS.md` shows endpoint ❌.
- **Must not** send `job_id` or `gpu_device` on `POST /jobs`.
- **Must not** add TypeScript types that are not in `packages/contracts/typescript/`.
- If a screen needs a field not in schema: **stop** and add schema + fixture first (or file a note in PR description listing the contract diff).

### Engine (`src/viana/`)

- **Must not** add HTTP handlers or FastAPI imports.
- CSV columns **must** match `events_raw.schema.json` / `events_15min.schema.json`.
- Disk artifacts (`checkpoint`, `run_result`) **must** match their schemas.

### API (`src/orchestrator/`)

- **Must not** embed CV logic; spawn `python -m viana` only.
- Routes **must** match `docs/api_contracts.md` and JSON schemas.
- Assigns `job_id`, `gpu_device`, `output_dir` — never accepts them from client on submit.

---

## Handoff checklist (contract change PR)

- [ ] JSON schema updated with `$id` and required fields
- [ ] TypeScript types match schema
- [ ] At least one fixture JSON for UI mock (for HTTP responses)
- [ ] `docs/api_contracts.md` updated
- [ ] `openapi.yaml` updated (if HTTP)
- [ ] Pydantic models updated (if job payload)
- [ ] `docs/PROJECT_STATUS.md` API matrix updated
- [ ] Other tracks notified via PR title/body: `contract: <what changed>`

---

## Conflict resolution

| Situation | Resolution |
|-----------|------------|
| UI and API disagree on field name | Schema in `packages/contracts/` wins; fix implementation |
| Engine CSV column not in schema | Add to `events_*.schema.json` before writing CSV |
| Doc says field exists, schema does not | Schema wins; fix doc |
| Endpoint marked ✅ but no implementation | Mark ❌ in `PROJECT_STATUS.md` until implemented |

See also: `docs/governance/SOURCE_OF_TRUTH.md`, `docs/governance/AI_SDLC.md` §4.
