#!/bin/bash
#
# Partition the dataset in Dataset/ into 4 train/val splits for MLFF1-4.
#
# Usage:
#   cd au_workflow/pzc/Training/
#   bash partition_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/baseline_mace_env/bin/python"

$PYTHON "${SCRIPT_DIR}/partition_dataset.py"
