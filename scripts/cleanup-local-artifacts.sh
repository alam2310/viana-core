#!/usr/bin/env bash
# Remove gitignored artifact folders at repo root (often created by Docker as user nobody).
# Run from repo root: ./scripts/cleanup-local-artifacts.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGETS=(debug_pretrain runs)
for d in "${TARGETS[@]}"; do
  if [[ -d "$d" ]]; then
    echo "Removing $d/ ..."
    rm -rf "$d" 2>/dev/null || sudo rm -rf "$d"
  fi
done

# Stale pytest cache from legacy tests
find tests -name 'test_classifier*.pyc' -delete 2>/dev/null || \
  sudo find tests -name 'test_classifier*.pyc' -delete 2>/dev/null || true

echo "Done."
