#!/bin/bash
#SBATCH --job-name=mace_au_slab
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=mace_au_slab_output_%j.txt

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${PROJECT_DIR}/src/inference_baseline.py" ]]; then
	PROJECT_DIR="/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests"
fi

if [[ ! -f "${PROJECT_DIR}/src/inference_baseline.py" ]]; then
	echo "[$(date)] ERROR: could not locate src/inference_baseline.py"
	exit 1
fi

cd "${PROJECT_DIR}"

echo "[$(date)] Starting MACE Au slab Job on Unity..."

module purge
module load cuda/12.1
module load conda/latest

conda activate /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/baseline_mace_env

export UV_CACHE_DIR="/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/baseline_mace_env/.cache/uv"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONWARNINGS="ignore::FutureWarning:e3nn\\..*,ignore::FutureWarning:mace\\..*,ignore::UserWarning:mace\\..*"

echo "[$(date)] Working directory: $PWD"
echo "[$(date)] Python: $(which python)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

echo "[$(date)] Running src/inference_baseline.py (Au slab only)..."
python -u "${PROJECT_DIR}/src/inference_baseline.py" \
    --structure "${PROJECT_DIR}/data/input_structures/au-slab-only.extxyz" \
    --log_interval 50 \
    --results_dir "${PROJECT_DIR}/data/results/au_slab"

LATEST_RUN_FILE="${PROJECT_DIR}/data/results/au_slab/latest_run.txt"
SLURM_STDOUT_FILE="${PROJECT_DIR}/scripts/mace_au_slab_output_${SLURM_JOB_ID:-}.txt"
if [[ -f "${LATEST_RUN_FILE}" ]]; then
	RUN_DIR="$(tr -d '\r\n' < "${LATEST_RUN_FILE}")"
	if [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
		if [[ -f "${SLURM_STDOUT_FILE}" ]]; then
			cp "${SLURM_STDOUT_FILE}" "${RUN_DIR}/"
			echo "[$(date)] Copied Slurm output to ${RUN_DIR}"
		fi
	fi
fi

echo "[$(date)] Job Completed."
