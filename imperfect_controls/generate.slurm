#!/bin/bash
# Baseline: Fill50K test samples with *clean* control hints (plain Fill50KDataset).
# Submit from repo root:  sbatch imperfect_controls/generate.slurm
#
# Override checkpoint (trained on clean hints), e.g.:
#   CKPT=imperfect_controls/checkpoints/clean/last.ckpt sbatch imperfect_controls/generate.slurm

#SBATCH -p mit_normal_gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J generate_fill50k
#SBATCH -o generate_fill50k_%j.out
#SBATCH -e generate_fill50k_%j.err

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

CKPT="${CKPT:-imperfect_controls/checkpoints/clean/last.ckpt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT=/path/to/clean-hint_training.ckpt before sbatch." >&2
  exit 1
fi

echo "[$(date)] Starting baseline generate job ${SLURM_JOB_ID} in $(pwd)"
echo "Using checkpoint: ${CKPT}"

python imperfect_controls/generate.py \
  --checkpoint "${CKPT}" \
  --num-samples 50 \
  --output-dir imperfect_controls/generated/baseline_slurm

echo "[$(date)] Generate job complete."
