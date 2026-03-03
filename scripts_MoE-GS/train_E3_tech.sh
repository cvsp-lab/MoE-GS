#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCENES=("Birthday" "Fabien" "Painter" "Theater" "Train")
ITERATIONS=(2000 5000)

BASE_CONFIG="configs/techni"
BASE_DATASET="/home/mhj/database2/ETRI/dataset/technicolor"

EX4DGS_PATH="/home/mhj/database2/ETRI/Pretrained/technicolor/Ex4DGS_2"
ED3DGS_PATH="/home/mhj/database2/ETRI/Pretrained/technicolor/E-D3DGS"
STG_PATH="/home/mhj/database2/ETRI/Pretrained/technicolor/STG_shs"

SAVE_PATH="/home/mhj/test/sample_data/Technicolor"

for SCENE in "${SCENES[@]}"; do
    # echo "Training scene: $SCENE"
    # python train_E3_tech.py --config "${BASE_CONFIG}/${SCENE}.json" \
    #     --source_path "${BASE_DATASET}/${SCENE}" \
    #     --model_path "${EX4DGS_PATH}/${SCENE}" \
    #     --emb_path "${ED3DGS_PATH}/${SCENE}" \
    #     --stg_path "${STG_PATH}/${SCENE}" \
    #     --save_path "${SAVE_PATH}/${SCENE}"

    for ITER in "${ITERATIONS[@]}"; do
        python render_E3_tech.py --skip_train \
            --source_path "${BASE_DATASET}/${SCENE}" \
            --save_path "${SAVE_PATH}/${SCENE}" \
            --iteration $ITER
    done
done