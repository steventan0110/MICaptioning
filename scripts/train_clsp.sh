#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -wd /home/czhan105/MICaptioning/MICaptioning-2
#$ -N MIC-Training
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M czhan105@jhu.edu
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
WORK_DIR=/home/czhan105/MICaptioning/MICaptioning-2
TRAIN=$WORK_DIR/train.py
DATA=/export/b02/wtan/MIC-dataset/datasets
CHECKPOINT_FOLDER=/export/b02/czhan105/MIC-checkpoints/transformer
CAPTION=/export/b02/wtan/MIC-dataset/iu_xray_captions.json

conda activate MIC

python $TRAIN \
        --mode train \
        --caption-dir $CAPTION \
        --data-dir $DATA \
        --arch transformer \
        --max-epoch 100 \
        --log-interval 5 \
        --batch-size 32 \
        -lr 0.01 \
        --save-dir $CHECKPOINT_FOLDER \
