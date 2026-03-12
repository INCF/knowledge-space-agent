#!/usr/bin/env bash
# Verify Milestone 1 and 2 locally. No GCP credentials required.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Milestone 1: TTFR module and demo ==="
cd "$ROOT/backend"
python3 -c "
from ttfr_estimator import estimate_ttfr
e = estimate_ttfr(datasource_id='scr_005031_openneuro')
assert e.summary.min_days >= 0
print('  Import and estimate_ttfr: OK')
print('  Sample:', e.summary)
"
python3 demo_ttfr.py | head -20
echo "  Demo: OK (output above)"

echo ""
echo "=== Milestone 2: Tests in tests/ ==="
cd "$ROOT"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -25

echo ""
echo "=== Done. No GCP credentials required. ==="
