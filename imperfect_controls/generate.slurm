#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
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

echo "[$(date)] Starting generate job ${SLURM_JOB_ID} in $(pwd)"
python imperfect_controls/generate.py \
  --checkpoint imperfect_controls/checkpoints/testrun1/last.ckpt \
  --num-samples 5
echo "[$(date)] Generate job complete."
