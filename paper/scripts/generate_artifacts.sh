#!/usr/bin/env bash
# Generates paper items 1, 2, and 6 (accuracy/AUC table, confusion matrices,
# failure examples) for one detector from its predictions CSVs.
# Items 3 (score distributions) and 4/5 (spectra, highpass energy) are
# generator-input-only (not detector-specific unless preprocessing differs)
# and are produced separately.
#
# Usage:
#   bash generate_artifacts.sh <detector_dir_name> <detector_display_name> \
#     BigGan=/path/predictions_BigGan_full.csv Glide=/path/... flux=... sd3=... sdxl=...
set -e
DETECTOR_DIR="$1"; shift
DETECTOR_NAME="$1"; shift
CSV_ARGS=("$@")

PAPER_ROOT="/home/student/data/paper"
OUT_DIR="${PAPER_ROOT}/${DETECTOR_DIR}"
SCRIPTS="${PAPER_ROOT}/scripts"

source /home/student/qwen_env/bin/activate

echo "=== Accuracy/AUC table + confusion matrices ==="
python3 "${SCRIPTS}/compute_metrics_table.py" \
  --csv "${CSV_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --detector_name "$DETECTOR_NAME"

echo "=== Score distribution histograms ==="
python3 /home/student/data/UniversalFakeDetect/plot_score_distribution.py \
  --csv "${CSV_ARGS[@]}" \
  --out "${OUT_DIR}/score_distributions/score_dist_all.png"

echo "=== Failure examples ==="
for entry in "${CSV_ARGS[@]}"; do
  label="${entry%%=*}"
  path="${entry#*=}"
  python3 "${SCRIPTS}/failure_examples.py" \
    --csv "$path" --label "$label" --n 6 \
    --out_dir "${OUT_DIR}/failure_examples"
done

echo "=== Done: artifacts written under ${OUT_DIR} ==="
