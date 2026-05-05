#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J eval_fill50k
#SBATCH -o eval_fill50k_%j.out
#SBATCH -e eval_fill50k_%j.err

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

echo "[$(date)] Starting evaluate job ${SLURM_JOB_ID} in $(pwd)"
python imperfect_controls/evaluate.py \
  --checkpoint imperfect_controls/checkpoints/blur_finetune/imperfect50-p=0.20-step-best-step=000499.ckpt \
  --output-csv imperfect_controls/eval_results/blur_finetune.csv \
  --output-summary-csv imperfect_controls/eval_results/blur_finetune_summary.csv \
  --max-samples 10 \
  --debug-mask-dir imperfect_controls/debug_masks/blur_finetune
echo "[$(date)] Evaluate job complete."
