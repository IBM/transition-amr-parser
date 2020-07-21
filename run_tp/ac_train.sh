#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### CONFIG
dir=$(dirname $0)
if [ -z "$1" ]; then
    config="config.sh"
else
    config=$1
fi
. $dir/$config    # we should always call from one level up

##### script specific config
max_epoch=100
seed=42

##### TRAINING
rm -Rf $MODEL_FOLDER
# python -m ipdb fairseq_ext/train.py \
python fairseq_ext/train.py \
    $DATA_FOLDER \
    --user-dir ../fairseq_ext \
    --task amr_pointer \
    --max-epoch $max_epoch \
    --arch transformer_pointer \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --pretrained-embed-dim $PRETRAINED_EMBED_DIM \
    --lr 0.0005 \
    --min-lr 1e-09 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy_pointer \
    --label-smoothing 0.01 \
    --loss-coef 1 \
    --keep-last-epochs 40 \
    --max-tokens 3584 \
    --log-format json \
    --seed $seed \
    --save-dir $MODEL_FOLDER
