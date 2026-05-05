#!/bin/bash
# Baseline: Fill50K test samples with *clean* control hints (plain Fill50KDataset).
# Submit from repo root:  sbatch imperfect_controls/generate.slurm

#SBATCH -p mit_preemptable
#SBATCH -c 6
#SBATCH -G 1
#SBATCH --mem=96G
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

echo "[$(date)] Starting baseline generate job ${SLURM_JOB_ID} in $(pwd)"

python imperfect_controls/generate.py \
  --checkpoint imperfect_controls/checkpoints/clean/last.ckpt \
  --num-samples 10 \
  --output-dir imperfect_controls/generated/clean_noise_patchy_strong \
  --corrupt-fraction 1 \
  --corruption-type "noise_patchy_strong"

echo "[$(date)] Generate job complete."
