#!/bin/bash
# Train ControlNet on Fill50K with disjoint corrupted controls (edge_segment_remove).
# Checkpoints: imperfect_controls/checkpoints/imperfect50_edge_drop/

#SBATCH -p mit_normal_gpu
#SBATCH -c 6
#SBATCH -G 2
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J train_imperfect50
#SBATCH -o train_imperfect50_%j.out
#SBATCH -e train_imperfect50_%j.err

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

echo "[$(date)] Starting imperfect-controls train job ${SLURM_JOB_ID} in $(pwd)"
python imperfect_controls/train.py \
  --corrupt-fraction 0.5 \
  --corruption-type edge_segment_remove \
  --max-steps 8000 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --num-gpus 2 \
  --run-name imperfect50_edge_drop
echo "[$(date)] Train job complete."
