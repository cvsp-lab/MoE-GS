#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# SCENES=("coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_salmon_1" "flame_steak" "sear_steak")
SCENES=("flame_salmon_1" "flame_steak" "sear_steak")
ITERATIONS=(2000 5000)

BASE_CONFIG="configs/N3V"
BASE_DATASET="/home/mhj/database2/ETRI/dataset/n3v"

EX4DGS_PATH="/home/mhj/database2/ETRI/Pretrained/n3v/Ex4DGS"
ED3DGS_PATH="/home/mhj/database2/ETRI/Pretrained/n3v/E-D3DGS"
FDGS_PATH="/home/mhj/database2/ETRI/Pretrained/n3v/4DGS"
STG_PATH="/home/mhj/database2/ETRI/Pretrained/n3v/STG"
GS4D_PATH="/home/mhj/database2/ETRI/Pretrained/n3v/4DGaussians"

SAVE_PATH="/home/mhj/database2/ETRI/final_25/E3"

for SCENE in "${SCENES[@]}"; do
    echo "Training scene: $SCENE"
    python train_E3.py --config "${BASE_CONFIG}/${SCENE}.json" \
        --source_path "${BASE_DATASET}/${SCENE}" \
        --model_path "${EX4DGS_PATH}/${SCENE}" \
        --emb_path "${ED3DGS_PATH}/${SCENE}" \
        --fgaussian_path "${GS4D_PATH}/${SCENE}" \
        --save_path "${SAVE_PATH}/${SCENE}"

    for ITER in "${ITERATIONS[@]}"; do
        python render_E3.py --skip_train \
            --source_path "${BASE_DATASET}/${SCENE}" \
            --save_path "${SAVE_PATH}/${SCENE}" \
            --iteration $ITER
    done
done