#!/bin/bash
DATA_DIR="/USER-DEFINED-PATH/Dataset/"
MODEL_PATH="train_logs/clust/model_best.pth.tar"
TARGET="market1501"         # market1501  dukemtmc  msmt17  lab314  mta
ARCH="osnet_ain_x0_5"		# resnet101  resnet_ibn101a  osnet_x1_0  osnet_ain_x1_0
							# resnet50  resnet_ibn50a  osnet_x0_5  osnet_ain_x0_5


export PYTHONPATH=$PYTHONPATH:`pwd`

if [[ "$ARCH" != "osnet_ain_x0_5" && "$ARCH" != "osnet_ain_x1_0" && \
      "$ARCH" != "osnet_x0_5" && "$ARCH" != "osnet_x1_0" ]] ; then
	FEATURES=2048
else
	FEATURES=512
fi

CUDA_VISIBLE_DEVICES=0 \
python test/test_model.py -dt ${TARGET} --arch ${ARCH} \
	--height 256 --width 128 --features ${FEATURES} \
	--data-dir ${DATA_DIR} \
	--resume ${MODEL_PATH}
