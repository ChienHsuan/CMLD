#!/bin/bash
DATA_DIR="/USER-DEFINED-PATH/Dataset/"
SOURCE="dukemtmc"             # market1501  dukemtmc
TARGET="market1501"
ARCH="resnet50"               # resnet101  resnet_ibn101a  resnet50  resnet_ibn50a
SEED=0


export PYTHONPATH=$PYTHONPATH:`pwd`

if [[ "$ARCH" != "osnet_ain_x0_5" && "$ARCH" != "osnet_ain_x1_0" && \
      "$ARCH" != "osnet_x0_5" && "$ARCH" != "osnet_x1_0" ]] ; then
	FEATURES=0
else
	FEATURES=512
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/pretrain.py \
    -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${SEED} --margin 0.0 \
    --num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 40 \
    --features ${FEATURES} \
    --data-dir ${DATA_DIR} \
    --logs-dir pretrain_logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-${SEED} \
    --print-freq 50
