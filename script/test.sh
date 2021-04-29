WORK_DIR=/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning
TRAIN=$WORK_DIR/train.py
DATA=$WORK_DIR/datasets
CHECKPOINT_FOLDER=$WORK_DIR
CAPTION=$WORK_DIR/iu_xray/iu_xray_captions.json

python $TRAIN \
    --mode test --cpu \
    --caption-dir $CAPTION \
    --data-dir $DATA \
    --max-epoch 10 \
    --load-dir $CHECKPOINT_FOLDER/checkpoint95.pt