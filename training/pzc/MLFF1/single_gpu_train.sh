#!/bin/bash
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-node=1    # Number of MPI ranks per node
#SBATCH --gres=gpu:1       # Number of requested gpus per node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time 48:00:00        # Walltime, format: HH:MM:SS
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --job-name='train_1_run'
#SBATCH --account=pi_ozguryilmaze_umass_edu
#SBATCH --chdir=/scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/aravadikeshr_tests/au_workflow/pzc/Training/MLFF1
#SBATCH --constraint="vram40&bf16"

# Source environment for Unity HPC
module purge
module load cuda/12.1
module load conda/latest
conda activate /scratch/workspace/anshitagupta_umass_edu-ai-gold-confinement/conda_envs/baseline_mace_env
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU info
nvidia-smi

START_TIME=$(date +%s)

# Dataset pre-processing (required for Multi-GPU training)
rm -rf processed_data
mkdir -p processed_data

python -m mace.cli.preprocess_data \
    --train_file="train_model_1.xyz" \
    --valid_file="val_model_1.xyz"\
	--energy_key='energy'\
	--forces_key='forces'\
    --atomic_numbers="[1, 8, 79]" \
    --r_max=6 \
    --h5_prefix="processed_data/" \
    --compute_statistics \
    --E0s="average" \
    --seed=558 \

# Fix numpy 2.0 serialization issue: strip np.float64(...) wrappers from statistics.json
sed -i 's/np\.float64(\([^)]*\))/\1/g' processed_data/statistics.json

# Training
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_GPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv --loop=5 > gpu_stats.csv &
NVIDIASMI_PID=$!

python -W "ignore::UserWarning:torch.jit._check" -m mace.cli.run_train \
    --name="MACE_Au_H2O_PZC_1_run" \
    --train_file="./processed_data/train" \
	--valid_file="./processed_data/val" \
	--statistics_file="./processed_data/statistics.json" \
    --energy_key='energy'\
	--forces_key='forces'\
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --r_max=6 \
    --max_L=0 \
	--num_channels=256 \
    --batch_size=16 \
    --max_num_epochs=800 \
    --valid_batch_size=16 \
    --patience=10 \
    --eval_interval=5 \
    --swa \
    --start_swa=320 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --seed=300 \
    --default_dtype="float32"\
	--num_workers=8\
    --save_cpu

kill $NVIDIASMI_PID

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf "Wall clock time: %02d:%02d:%02d (hh:mm:ss)\n" \
    $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) $(( ELAPSED % 60 ))
