WORK_DIR=/home/czhan105/MICaptioning
TRAIN=$WORK_DIR/train.py
DATA=/export/b02/wtan/MIC-dataset/datasets
CHECKPOINT_FOLDER=/export/b02/czhan105/MIC-checkpoints/chexnet-lstm
CAPTION=/export/b02/wtan/MIC-dataset/iu_xray_captions.json

python $TRAIN \
    --mode test --cpu \
    --caption-dir $CAPTION \
    --data-dir $DATA \
    --max-epoch 10 \
    --load-dir $CHECKPOINT_FOLDER/checkpoint95.pt \
    --encoder-arch chexnet
