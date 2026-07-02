#!/bin/bash
#SBATCH --job-name=drifting_train_latent_B_4gpu
#SBATCH --partition=blackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/playpen-shared/dinara/drifting/runs/latent_B_4gpu_full/slurm_%j.log
#SBATCH --error=/playpen-shared/dinara/drifting/runs/latent_B_4gpu_full/slurm_%j.err

# Setup
source /playpen-shared/dinara/miniconda3/etc/profile.d/conda.sh
conda activate drifting-gpu

export JAX_PLATFORMS=cuda,cpu
export JAX_COMPILATION_CACHE_DIR=/playpen-shared/dinara/.jax_cache
export TMPDIR=/playpen-shared/dinara/tmp
export WANDB_API_KEY=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

cd /playpen-shared/dinara/drifting

mkdir -p runs/latent_B_4gpu_full

python main.py --gen \
    --config configs/gen/latent_sota_B_4gpu_full_train.yaml \
    --workdir runs/latent_B_4gpu_full \
    2>&1 | tee -a runs/latent_B_4gpu_full/train.log
