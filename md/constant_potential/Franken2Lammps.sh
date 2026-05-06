# Convert a Franken-trained MACE model to LAMMPS format (.model-lammps.pt)
# Usage: bash Franken2Lammps.sh <conda_env_path> <model_file.model>
# Example: bash Franken2Lammps.sh /path/to/franken_env results_multi_potential/case_m05_RF_512/best_model.model

source ~/.bashrc

# Activate the conda environment with MACE/Franken installed
ENV_PATH=$1
mamba activate "$ENV_PATH"

# Convert the model to LAMMPS format using MACE's compile utility
# (Franken outputs a standard MACE .model file, compilable with the same tool)
python $ENV_PATH/lib/python3.*/site-packages/mace/cli/create_lammps_model.py $2

mamba deactivate

echo "LAMMPS model created: ${2}-lammps.pt"
