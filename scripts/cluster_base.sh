#!/bin/bash
DATA_DIR="/USER-DEFINED-PATH/Dataset/"
SOURCE="dukemtmc"					# market1501  dukemtmc  msmt17  lab314  mta
TARGET="market1501"
ARCH_TEACHER="osnet_ain_x0_5"		# resnet101  resnet_ibn101a  osnet_x1_0  osnet_ain_x1_0
									# resnet50  resnet_ibn50a  osnet_x0_5  osnet_ain_x0_5


export PYTHONPATH=$PYTHONPATH:`pwd`

if [[ "$TARGET" == "msmt17" || "$TARGET" == "mta" ]] ; then
	RHO=0.0007
	ITERS=2400
else
	RHO=0.0016
	ITERS=1200
fi

if [[ "$ARCH_TEACHER" != "osnet_ain_x0_5" && "$ARCH_TEACHER" != "osnet_ain_x1_0" && \
      "$ARCH_TEACHER" != "osnet_x0_5" && "$ARCH_TEACHER" != "osnet_x1_0" ]] ; then
	FEATURES=2048
	INIT_T1=/USER-DEFINED-PATH/models/${SOURCE}TO${TARGET}/${ARCH_TEACHER}-pretrain-0/model_best.pth.tar
else
	FEATURES=512
	INIT_T1=/USER-DEFINED-PATH/models/${ARCH_TEACHER}/${SOURCE}/0/model.pth.tar-100
fi

CUDA_VISIBLE_DEVICES=0 \
python examples/cluster_base.py -ds ${SOURCE} -dt ${TARGET} --arch-teacher ${ARCH_TEACHER} \
	--num-instances 4 --lr 0.0001 --iters ${ITERS} -b 16 --epochs 40 \
	--height 256 --width 128 --features ${FEATURES} \
	--rho ${RHO} \
	--init-t1 ${INIT_T1} \
	--data-dir ${DATA_DIR} \
	--logs-dir train_logs/clust_${SOURCE}-2-${TARGET}_${ARCH_TEACHER}/ \
	--eval-step 1 \
	--print-freq 100
