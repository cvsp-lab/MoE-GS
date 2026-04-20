# Third-Party Dependencies Recovery Guide

This project depends on several external open-source components that are not included in this repository.
Follow the instructions below to restore them before building or running the code.

## thirdparty/

### embedding (E-D3DGS)
```bash
# Source: https://github.com/JeongminB/E-D3DGS
# Copy the following directories from the E-D3DGS repository:
#   - scene/          -> thirdparty/embedding/scene/
#   - utils/          -> thirdparty/embedding/utils/
#   - arguments/      -> thirdparty/embedding/arguments/
#   - diff-gaussian-rasterization/ -> thirdparty/embedding/diff-gaussian-rasterization/
git clone https://github.com/JeongminB/E-D3DGS.git /tmp/E-D3DGS
cp -r /tmp/E-D3DGS/scene /tmp/E-D3DGS/utils /tmp/E-D3DGS/arguments thirdparty/embedding/
cp -r /tmp/E-D3DGS/submodules/diff-gaussian-rasterization thirdparty/embedding/diff-gaussian-rasterization
```

### Hexplane (4DGaussians)
```bash
# Source: https://github.com/hustvl/4DGaussians
# Contains: scene/, utils/, arguments/, depth-diff-gaussian-rasterization/
git clone https://github.com/hustvl/4DGaussians.git /tmp/4DGaussians
cp -r /tmp/4DGaussians/scene thirdparty/Hexplane/scene
cp -r /tmp/4DGaussians/utils thirdparty/Hexplane/utils
cp -r /tmp/4DGaussians/arguments thirdparty/Hexplane/arguments
cp -r /tmp/4DGaussians/submodules/depth-diff-gaussian-rasterization thirdparty/Hexplane/depth-diff-gaussian-rasterization
```

### polynomial (SpacetimeGaussians / STG)
```bash
# Source: https://github.com/oppo-us-research/SpacetimeGaussians
# Contains: scene/, scene_tech/, utils/, helper_model.py, helper_train.py
# Note: This directory contains modified versions adapted for MoE-GS.
#       Restore from project backup if available.
```

### reparmetrize
```bash
# Contains: scene/, utils/, arguments/, pointops2/, Fdiff_gaussian_rasterization/
# Note: Custom reparametrization module based on Ex4DGS.
#       Restore from project backup if available.
```

### weight (gaussian_rasterization_ch3)
```bash
# Source: Modified from https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth
# Contains: cuda_rasterizer/, diff_gaussian_rasterization_ch3/
# Note: 3-channel weight variant of the Gaussian rasterizer.
#       Restore from project backup if available.
```

## submodules/

### diff_gaussian_rasterization_df
```bash
# Source: Inria GRAPHDECO 3D Gaussian Splatting rasterizer (depth+flow variant)
# Original: https://github.com/graphdeco-inria/diff-gaussian-rasterization
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff_gaussian_rasterization_df
# Note: This is a modified version with depth and flow support.
#       The vanilla version may need manual patches. Restore from backup if available.
```

### simple-knn
```bash
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
```

## lpipsPyTorch/
```bash
# Source: https://github.com/richzhang/PerceptualSimilarity
# Contains: modules/lpips.py, modules/networks.py, modules/utils.py, __init__.py
pip install lpips
# Or copy the lpipsPyTorch module from the PerceptualSimilarity repository.
```

## Building CUDA Extensions

After restoring all dependencies, build the CUDA extensions:
```bash
# Each submodule with a setup.py needs to be installed:
pip install submodules/diff_gaussian_rasterization_df/
pip install submodules/simple-knn/
pip install thirdparty/embedding/diff-gaussian-rasterization/
pip install thirdparty/Hexplane/depth-diff-gaussian-rasterization/
pip install thirdparty/reparmetrize/pointops2/
pip install thirdparty/weight/gaussian_rasterization_ch3/
```

## glm (OpenGL Mathematics)

Each rasterization module requires glm in its `third_party/` directory:
```bash
git clone https://github.com/g-truc/glm.git
# Copy to each location that needs it:
#   thirdparty/embedding/diff-gaussian-rasterization/third_party/glm
#   thirdparty/Hexplane/depth-diff-gaussian-rasterization/third_party/glm
#   thirdparty/reparmetrize/Fdiff_gaussian_rasterization/third_party/glm
#   thirdparty/weight/gaussian_rasterization_ch3/third_party/glm
#   submodules/diff_gaussian_rasterization_df/third_party/glm
```
