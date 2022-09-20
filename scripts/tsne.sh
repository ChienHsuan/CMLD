#!/bin/bash
DATA_DIR="/USER-DEFINED-PATH/Dataset/"
MODEL_PATH="/USER-DEFINED-PATH/model_best.pth.tar"
SOURCE="dukemtmc"
TARGET="market1501"
ARCH="resnet50"
OUTPUT="result.pdf"
NUM_IDS=20
POINT_LABEL_TYPE="mean"
SEED=0


export PYTHONPATH=$PYTHONPATH:`pwd`

if [[ "$ARCH" != "osnet_ain_x0_5" && "$ARCH" != "osnet_ain_x1_0" && \
      "$ARCH" != "osnet_x0_5" && "$ARCH" != "osnet_x1_0" ]] ; then
	FEATURES=0
else
	FEATURES=512
fi

CUDA_VISIBLE_DEVICES=0 \
python test/tsne.py \
    -dt ${TARGET} \
    -a ${ARCH} \
    --features ${FEATURES} \
    --num-ids ${NUM_IDS} \
    --point_label_type ${POINT_LABEL_TYPE} \
    --seed ${SEED} \
    --data-dir ${DATA_DIR} \
    --resume ${MODEL_PATH} \
    --output ${OUTPUT}
