source /mnt/d/Github/MICaptioning/MIC/bin/activate
TRAIN=/mnt/d/Github/MICaptioning/train.py
CAPTION=/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json
DATA=/mnt/d/Github/MICaptioning/datasets
CHECKPOINT=/mnt/d/Github/MICaptioning/checkpoints
python $TRAIN \
    --mode train --cpu \
    --caption-dir $CAPTION \
    --data-dir $DATA \
    --max-epoch 10 \
    --save-dir $CHECKPOINT