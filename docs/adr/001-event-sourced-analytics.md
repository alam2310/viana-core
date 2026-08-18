# ADR 001: Event-Sourced Analytics

## Status

Accepted

## Context

# Legacy scripts and experiments — see `legacy/README.md` and `legacy/PARITY.md`.
# Parity reference: legacy/inference/inference_engine.py

## Decision

- Emit one row per crossing to `{stem}_events.csv`
- Build `{stem}_15min.csv` in a separate aggregation stage
- Clock-aligned 15-minute windows with zero-filled vehicle-class grid

## Consequences

- Aggregation can be re-run without re-inference
- OCR/time fixes do not require re-processing video
- Engine hot loop stays lean
