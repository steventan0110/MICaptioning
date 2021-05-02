#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -wd /export/b10/jzhan237/MICaptioning
#$ -N MIC-Interactive
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jzhan237@jhu.edu
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=20G,gpu=1,hostname=c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

# Activate any environments, call your script, etc
WORK_DIR=$PWD
TRAIN=$WORK_DIR/train.py
DATA=/export/b02/wtan/MIC-dataset/datasets
CAPTION=/export/b02/wtan/MIC-dataset/iu_xray_captions.json

# Model to evaluate
MODEL_PATH=/export/b10/jzhan237/MICCheckpoints/checkpoint9.pt

conda activate MIC

python $TRAIN \
	-m interactive \
	--caption-dir $CAPTION \
	--data-dir $DATA \
	--load-dir $MODEL_PATH \
	--cpu
