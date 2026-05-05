#!/bin/bash
# Train ControlNet on Fill50K.

#SBATCH -p mit_preemptable
#SBATCH -c 6
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -J train_imperfect
#SBATCH -o train_imperfect_%j.out
#SBATCH -e train_imperfect_%j.err

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

echo "Waiting for 1 CUDA devices to become visible..."
for i in {1..60}; do
  visible_count=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
  echo "Attempt $i: visible CUDA devices = ${visible_count}"

  if [[ "${visible_count}" -ge 1 ]]; then
    echo "Found ${visible_count} CUDA devices."
    break
  fi

  sleep 5
done

if [[ "${visible_count}" -lt 1 ]]; then
  echo "ERROR: Expected 1 CUDA devices, but only ${visible_count} are visible after waiting."
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi || true
  exit 1
fi

echo "[$(date)] Starting imperfect-controls train job ${SLURM_JOB_ID} in $(pwd)"
python imperfect_controls/train.py \
  --corrupt-fraction 0.2 \
  --corruption-type noise \
  --max-steps 500 \
  --batch-size 4 \
  --effective-batch-size 16 \
  --learning-rate 5e-6 \
  --num-gpus 1 \
  --run-name noise_finetune \
  --finetune-from imperfect_controls/checkpoints/clean/last.ckpt \
  --val-every-n-steps 100 \
  --weights-save-every-n-steps 100
echo "[$(date)] Train job complete."
