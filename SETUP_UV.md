# MoE-GS: uv Environment Setup Guide

## Prerequisites

* CUDA Toolkit 11.8 or higher
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Environment Creation and Package Installation

```bash
# 1. Create a Python 3.9 virtual environment
uv venv --python 3.9

# 2. Activate
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -e .

# Or explicitly specify the PyTorch CUDA version:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .
```

## Build CUDA Extensions (after restoring third-party modules)

Refer to [`THIRDPARTY_RECOVERY.md`](THIRDPARTY_RECOVERY.md) for how to restore third-party modules.

```bash
# Build each submodule
uv pip install submodules/diff_gaussian_rasterization_df/
uv pip install submodules/simple-knn/
uv pip install thirdparty/embedding/diff-gaussian-rasterization/
uv pip install thirdparty/Hexplane/depth-diff-gaussian-rasterization/
uv pip install thirdparty/reparmetrize/pointops2/
uv pip install thirdparty/weight/gaussian_rasterization_ch3/
```

## Migration from conda

The existing `environment.yml` is retained for reference.
Using uv instead of conda provides faster environment creation and better reproducibility.
