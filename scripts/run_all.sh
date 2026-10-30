#!/usr/bin/env bash
# 전체 파이프라인. 사용법: scripts/run_all.sh [v1.0-mini|v1.0-trainval]
set -euo pipefail
cd "$(dirname "$0")/.."
V="${1:-v1.0-trainval}"
PY=.venv/bin/python
$PY scripts/01_build_index.py --version "$V"
$PY scripts/01b_align_check.py --version "$V"
$PY scripts/02_detect.py --version "$V"          # GPU, 이미 있는 scene 은 건너뜀
$PY scripts/03_tracks.py --version "$V"
$PY scripts/04_calibrate.py --version "$V"
$PY scripts/05_fuse.py --version "$V"
for s in val test; do $PY scripts/06_eval_ttc.py --version "$V" --split "$s"; done
$PY scripts/06b_sensitivity.py --version "$V"
$PY scripts/07_near_miss.py --version "$V"
$PY scripts/08_risk.py --version "$V"
