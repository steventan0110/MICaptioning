WORK_DIR=/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning

TRAIN=$WORK_DIR/train.py
DATA=$WORK_DIR/datasets
CHECKPOINT_FOLDER=$WORK_DIR/checkpoints
CAPTION=$WORK_DIR/iu_xray/iu_xray_captions.json

python $TRAIN \
    --mode train --cpu \
    --arch transformer \
    --caption-dir $CAPTION \
    --data-dir $DATA \
    --max-epoch 10 \


