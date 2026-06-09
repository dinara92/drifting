"""Global paths for the public Drift release."""

from __future__ import annotations

import os

# IMAGENET_PATH = "/path/to/imagenet"
# IMAGENET_CACHE_PATH = ""
# IMAGENET_FID_NPZ = "/path/to/imagenet_256_fid_stats.npz"
# IMAGENET_PR_NPZ = "/path/to/imagenet_val_prc_arr0.npz"

# HF_REPO_ID = "Goodeat/drifting"
# HF_ROOT = os.environ.get("HF_ROOT", "/path/to/hf_cache")

IMAGENET_PATH = "/playpen-shared/dinara/datasets/imagenet"
IMAGENET_CACHE_PATH = ""
IMAGENET_FID_NPZ = "/playpen-shared/dinara/datasets/imagenet/stats/imagenet_256_fid_stats.npz"
IMAGENET_PR_NPZ = "/playpen-shared/dinara/datasets/imagenet/stats/imagenet_val_prc_arr0.npz"

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", "/playpen-shared/dinara/.hf-cache/hub")
