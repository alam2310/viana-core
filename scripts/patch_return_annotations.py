#!/usr/bin/env python3
"""Add `-> None` return annotations to legacy/test Python functions missing any annotation."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("legacy/", "tests/")


def patch_file(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    if not rel.startswith(PREFIXES) and not rel.endswith("__init__.py"):
        return False
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_ann = node.returns is not None or any(
            a.annotation is not None for a in node.args.args
        )
        if has_ann:
            continue
        idx = node.lineno - 1
        line = lines[idx]
        if "->" in line or not line.rstrip().endswith(":"):
            continue
        lines[idx] = line.rstrip()[:-1] + " -> None:\n"
        changed = True
    if changed:
        try:
            path.write_text("".join(lines), encoding="utf-8")
        except OSError as exc:
            print(f"skip {path}: {exc}")
            return False
    return changed


def main() -> None:
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    count = 0
    for rel in files:
        path = ROOT / rel
        if patch_file(path):
            count += 1
            print(rel)
    print(f"patched {count} files")


if __name__ == "__main__":
    main()
