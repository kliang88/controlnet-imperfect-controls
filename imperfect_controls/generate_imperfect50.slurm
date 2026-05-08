#!/bin/bash
# Imperfect50 edge-drop: test samples with corrupted hints only (--only-corrupted).
# Uses same --corrupt-fraction 0.5 disjoint split as training; skips clean test rows.
# Default checkpoint: train_imperfect.slurm --run-name imperfect50_edge_drop
# Submit from repo root:  sbatch imperfect_controls/generate_imperfect50.slurm
#
# Optional:  CKPT=/path/to.ckpt sbatch imperfect_controls/generate_imperfect50.slurm

#SBATCH -p mit_normal_gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J gen_imperfect50_edge
#SBATCH -o generate_imperfect50_edge_drop_%j.out
#SBATCH -e generate_imperfect50_edge_drop_%j.err

set -euo pipefail

module load miniforge/25.11.0-0

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/imperfect_controls/train.py" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/train.py" && -f "${SLURM_SUBMIT_DIR}/../cldm/model.py" ]]; then
    cd "${SLURM_SUBMIT_DIR}/.."
  else
    exit 1
  fi
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

eval "$(conda shell.bash hook)"
conda activate control

CKPT="${CKPT:-imperfect_controls/checkpoints/imperfect50_edge_drop/last.ckpt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT=/path/to/your.ckpt before sbatch, or train with train_imperfect.slurm (--run-name imperfect50_edge_drop)." >&2
  exit 1
fi

echo "[$(date)] Starting imperfect50_edge_drop generate job ${SLURM_JOB_ID} in $(pwd)"
echo "Using checkpoint: ${CKPT}"

python imperfect_controls/generate.py \
  --checkpoint "${CKPT}" \
  --corruption-type edge_segment_remove \
  --num-samples 10 \
  --output-dir imperfect_controls/generated/imperfect50_edge_drop_slurm

echo "[$(date)] Generate job complete."
