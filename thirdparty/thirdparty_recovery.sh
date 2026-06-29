#!/bin/bash

mkdir thirdparty

mkdir thirdparty/embedding
git clone https://github.com/JeongminB/E-D3DGS.git /tmp/E-D3DGS
cp -r /tmp/E-D3DGS/scene /tmp/E-D3DGS/utils /tmp/E-D3DGS/arguments thirdparty/embedding/
cp -r /tmp/E-D3DGS/submodules/diff-gaussian-rasterization thirdparty/embedding/diff-gaussian-rasterization


mkdir thirdparty/Hexplane
git clone https://github.com/hustvl/4DGaussians.git /tmp/4DGaussians
cp -r /tmp/4DGaussians/scene thirdparty/Hexplane/scene
cp -r /tmp/4DGaussians/utils thirdparty/Hexplane/utils
cp -r /tmp/4DGaussians/arguments thirdparty/Hexplane/arguments
cp -r /tmp/4DGaussians/submodules/depth-diff-gaussian-rasterization thirdparty/Hexplane/depth-diff-gaussian-rasterization


# mkdir submodules
# git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff_gaussian_rasterization_df
# git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn


git clone https://github.com/g-truc/glm.git /tmp/glm
mkdir submodules/diff_gaussian_rasterization_df/third_party
mkdir thirdparty/embedding/diff-gaussian-rasterization/third_party
mkdir thirdparty/Hexplane/depth-diff-gaussian-rasterization/third_party
mkdir thirdparty/weight/gaussian_rasterization_ch3/third_party
mkdir thirdparty/reparmetrize/Fdiff_gaussian_rasterization/third_party
cp -r /tmp/glm submodules/diff_gaussian_rasterization_df/third_party/glm
cp -r /tmp/glm thirdparty/embedding/diff-gaussian-rasterization/third_party/glm
cp -r /tmp/glm thirdparty/Hexplane/depth-diff-gaussian-rasterization/third_party/glm
cp -r /tmp/glm thirdparty/weight/gaussian_rasterization_ch3/third_party/glm
cp -r /tmp/glm thirdparty/reparmetrize/Fdiff_gaussian_rasterization/third_party/glm
