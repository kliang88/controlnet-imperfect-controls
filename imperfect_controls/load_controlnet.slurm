#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J load_controlnet
#SBATCH -o load_controlnet_%j.out
#SBATCH -e load_controlnet_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/tool_add_control.py" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/../tool_add_control.py" ]]; then
    cd "${SLURM_SUBMIT_DIR}/.."
  else
    echo "Cannot find tool_add_control.py under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
  fi
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

eval "$(conda shell.bash hook)"
conda activate control

echo "[$(date)] Starting tool_add_control job ${SLURM_JOB_ID}"
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
echo "[$(date)] tool_add_control complete."
