#!/usr/bin/env bash
# Rebuild all results and figures from the public datasets (no model API calls).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
S=experiments/scripts

$PY -m pytest -q
$PY $S/download_data.py
$PY $S/prepare_data.py
$PY $S/compute_features.py

$PY $S/run_sprout.py                 # pool P6
$PY $S/run_sprout.py --lodo          # leave-one-domain-out
$PY $S/run_sprout.py --pool P13      # all 13 models
$PY $S/run_routerbench.py

for run in sprout_P6 sprout_P13 routerbench; do $PY $S/report.py --run "$run"; done
$PY $S/report_lodo.py
$PY $S/analyze_discrimination.py
$PY $S/fixed_fallback.py
$PY $S/measure_overhead.py
$PY $S/new_model_calibration.py
$PY $S/make_figures.py
