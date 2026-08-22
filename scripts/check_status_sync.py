#!/usr/bin/env python3
"""Verify STABILIZATION_BACKLOG, TRACKER mirror, and PROJECT_STATUS stay aligned."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs/steps/STABILIZATION_BACKLOG.md"
TRACKER = ROOT / "docs/steps/TRACKER.md"
PROJECT_STATUS = ROOT / "docs/PROJECT_STATUS.md"

SEQ_CELL = re.compile(r"^\s*(?:~~)?\*\*(S\d+)\*\*|^\s*(?:~~)?(S\d+)\b")


def normalize_status(cell: str) -> str:
    raw = cell.lower().strip("* ")
    if "parked" in raw:
        return "parked"
    if "open" in raw:
        return "open"
    if "deferred" in raw:
        return "deferred"
    if "wontfix" in raw:
        return "wontfix"
    if "fixed" in raw:
        return "fixed"
    return raw


def seq_from_first_cell(cell: str) -> str | None:
    m = SEQ_CELL.match(cell.strip())
    if not m:
        return None
    return m.group(1) or m.group(2)


def parse_backlog_table(text: str) -> dict[str, str]:
    idx = text.find("## Execution path")
    if idx < 0:
        raise ValueError("backlog ## Execution path not found")
    section = text[idx:]
    rows: dict[str, str] = {}
    in_table = False
    for line in section.splitlines():
        if line.startswith("| **S") or line.startswith("| ~~**S"):
            in_table = True
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        seq = seq_from_first_cell(cells[0])
        if not seq:
            continue
        rows[seq] = normalize_status(cells[-1])
    return rows


def parse_tracker_mirror(text: str) -> dict[str, str]:
    marker = "### Stabilization execution path"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError("TRACKER mirror heading not found")
    section = text[idx:]
    rows: dict[str, str] = {}
    in_table = False
    for line in section.splitlines():
        if line.startswith("| Seq |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("---"):
            break
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        seq = seq_from_first_cell(cells[0])
        if not seq:
            continue
        rows[seq] = normalize_status(cells[-1])
    return rows


def open_seq_from_backlog_summary(text: str) -> list[str]:
    m = re.search(r"open polish:\s*(.+?)\.", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return sorted(set(re.findall(r"S\d+", m.group(1))))


def open_seq_from_table(table: dict[str, str]) -> list[str]:
    return sorted(seq for seq, st in table.items() if st == "open")


def _leading_int(cell: str) -> int:
    m = re.match(r"(\d+)", cell.strip())
    if not m:
        raise ValueError(f"expected leading integer in summary cell: {cell!r}")
    return int(m.group(1))


def parse_summary_counts(text: str) -> tuple[int, int, int, int]:
    for line in text.splitlines():
        if not line.startswith("| 0 |"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        polish_open = _leading_int(cells[2])
        parked = _leading_int(cells[3])
        done_active = cells[4]
        m = re.match(r"\*\*(\d+)\s*/\s*(\d+)\*\*", done_active)
        if not m:
            raise ValueError(f"unexpected done/active cell: {done_active!r}")
        return polish_open, parked, int(m.group(1)), int(m.group(2))
    raise ValueError("backlog summary row not found")


def project_status_open_seqs(text: str) -> set[str]:
    m = re.search(r"Open Seq:.*?\(\*\*([^)]+)\*\*\)", text)
    if not m:
        return set()
    return set(re.findall(r"S\d+", m.group(1)))


def expand_open_seq_range(text: str) -> set[str]:
    """Expand S34–S36 or S34, S35, S36 style lists."""
    seqs: set[str] = set()
    for m in re.finditer(r"S(\d+)(?:\s*[–-]\s*S(\d+))?", text):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        for n in range(start, end + 1):
            seqs.add(f"S{n}")
    return seqs


def main() -> int:
    errors: list[str] = []

    backlog_text = BACKLOG.read_text(encoding="utf-8")
    tracker_text = TRACKER.read_text(encoding="utf-8")
    status_text = PROJECT_STATUS.read_text(encoding="utf-8")

    backlog_table = parse_backlog_table(backlog_text)
    tracker_table = parse_tracker_mirror(tracker_text)

    backlog_seqs = set(backlog_table)
    tracker_seqs = set(tracker_table)

    missing_in_tracker = sorted(backlog_seqs - tracker_seqs)
    extra_in_tracker = sorted(tracker_seqs - backlog_seqs)
    if missing_in_tracker:
        errors.append(
            f"TRACKER mirror missing Seq: {', '.join(missing_in_tracker)}"
        )
    if extra_in_tracker:
        errors.append(
            f"TRACKER mirror has extra Seq not in backlog: {', '.join(extra_in_tracker)}"
        )

    for seq in sorted(backlog_seqs & tracker_seqs):
        if backlog_table[seq] != tracker_table[seq]:
            errors.append(
                f"{seq} status mismatch: backlog={backlog_table[seq]!r} "
                f"tracker={tracker_table[seq]!r}"
            )

    open_backlog = open_seq_from_table(backlog_table)
    open_summary = open_seq_from_backlog_summary(backlog_text)
    if open_summary != open_backlog:
        errors.append(
            f"Backlog summary open polish {open_summary!r} != "
            f"table open rows {open_backlog!r}"
        )

    try:
        polish_open, parked, done, active = parse_summary_counts(backlog_text)
        expected_open = len(open_backlog)
        expected_parked = len(
            [s for s, st in backlog_table.items() if st == "parked"]
        )
        expected_done = len(
            [s for s, st in backlog_table.items() if st in ("fixed", "parked")]
        )
        expected_active = len(backlog_table)
        if polish_open != expected_open:
            errors.append(
                f"Summary polish open={polish_open} but table has {expected_open}"
            )
        if parked != expected_parked:
            errors.append(
                f"Summary parked={parked} but table has {expected_parked}"
            )
        if done != expected_done:
            errors.append(
                f"Summary done={done}/{active} but table has "
                f"{expected_done}/{expected_active}"
            )
        if active != expected_active:
            errors.append(
                f"Summary active total={active} but table has {expected_active} rows"
            )
    except ValueError as exc:
        errors.append(str(exc))

    ps_open = project_status_open_seqs(status_text)
    backlog_open_set = set(open_backlog)
    if ps_open:
        if ps_open != backlog_open_set:
            errors.append(
                f"PROJECT_STATUS Open Seq {sorted(ps_open)!r} != "
                f"backlog open {sorted(backlog_open_set)!r}"
            )
    elif backlog_open_set:
        errors.append(
            "PROJECT_STATUS missing Open Seq line but backlog has open rows"
        )

    if errors:
        print("Status sync check FAILED:\n", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nFix per docs/governance/STATUS_SYNC.md "
            "(backlog SoT → TRACKER mirror → PROJECT_STATUS).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Status sync OK: {len(backlog_table)} Seq rows; "
        f"open={open_backlog or 'none'}; done={done}/{active}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
