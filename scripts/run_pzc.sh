#!/bin/bash
#SBATCH --job-name=pzc_lammps
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a16
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=pzc_output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -d "${PROJECT_DIR}/workflows/pzc/MD" ]]; then
    PROJECT_DIR="/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests"
fi

cd "${PROJECT_DIR}/workflows/pzc/MD"

echo "[$(date)] Starting PZC LAMMPS workflow on Unity..."
echo "[$(date)] Working directory: $PWD"

module purge
module load cuda/12.1
module load conda/latest

conda activate /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/baseline_mace_env

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Run LAMMPS on the PZC input
echo "[$(date)] Running LAMMPS for PZC workflow..."
INPUT_FILE="MACE_Cu_111_H2O_PZC_lammps.in"
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "[$(date)] ERROR: ${INPUT_FILE} not found in $PWD"
    exit 1
fi

mpirun -np ${SLURM_NTASKS:-1} lmp_mpi -in "${INPUT_FILE}" 2>&1 | tee pzc_lammps_run.log

echo "[$(date)] PZC LAMMPS workflow finished."
