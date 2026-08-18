# ADR 002: Backend-Owned Job Management

## Status

Accepted

## Context

Job IDs and GPU assignment must not be managed by the browser.

## Decision

- FastAPI orchestrator owns queue, state machine, `job_id`, and `gpu_device`
- UI submits `JobSubmitRequest` without those fields
- Workers spawn engine via subprocess

## Consequences

- Safe concurrent processing on 2 GPUs
- UI is a thin client; queue syncs via `GET /jobs`
