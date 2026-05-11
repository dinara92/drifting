# drifting-gpu Environment Setup

Tested on: NVIDIA RTX A6000 × 8, CUDA 12.8, unites1

## Create environment

```bash
conda create -n drifting-gpu python=3.10 -y
conda activate drifting-gpu
```

## Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

## Install JAX (pinned — versions must all match)

```bash
pip install "jax==0.4.36" "jaxlib==0.4.36" "jax-cuda12-plugin==0.4.36"
```

> Do NOT let requirements.txt override these. Install JAX before or after,
> and re-pin if pip upgrades them.

## Install repo requirements

```bash
pip install -r requirements.txt

# Re-pin JAX in case requirements.txt changed it
pip install "jax==0.4.36" "jaxlib==0.4.36" "jax-cuda12-plugin==0.4.36"
```

## Environment variable

```bash
export JAX_PLATFORMS=cuda,cpu
```

Add to `~/.bashrc` or your Slurm job script to avoid setting it each session.

## Verify

```bash
python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0), ..., CudaDevice(id=7)]

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# True 8 NVIDIA RTX A6000
```