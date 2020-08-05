#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### check if the script is being sourced from other script or directly called
(return 0 2>/dev/null) && sourced=1 || sourced=0
# [[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "script ${BASH_SOURCE[0]} is being sourced ..." || echo "script ${BASH_SOURCE[0]} is NOT being sourced ..."
# [[ "${BASH_SOURCE[0]}" != "${0}" ]] && sourced=1 || sourced=0
# echo $sourced


##### CONFIG
if [[ $sourced == 0 ]]; then
    dir=$(dirname $0)
    if [ ! -z "${1+x}" ]; then
        config=$1
        . $dir/$config    # we should always call from one level up
    fi
    # NOTE: when the first configuration argument is not provided, this script must
    #       be called from other scripts
fi


##### script specific config
if [ -z ${max_epoch+x} ]; then
    max_epoch=100
fi
seed=${seed:-42}
# max_epoch=100
# seed=42


##### TRAINING
# rm -Rf $MODEL_FOLDER

if [ -d $MODEL_FOLDER ]; then
    
    echo "Directory to processed data $MODEL_FOLDER already exists --- do nothing."

else

    # python -m ipdb fairseq_ext/train.py \
    python fairseq_ext/train.py \
        $DATA_FOLDER \
        --emb-dir $EMB_FOLDER \
        --user-dir ../fairseq_ext \
        --task amr_action_pointer \
        --append-eos-to-target 0 \
        --collate-tgt-states 1 \
        --shift-pointer-value $shift_pointer_value \
        --apply-tgt-vocab-masks $tgt_vocab_masks \
        --share-decoder-input-output-embed $share_decoder_embed \
        --apply-tgt-src-align $apply_tgt_src_align \
        --tgt-src-align-focus $tgt_src_align_focus \
        \
        --pointer-dist-decoder-selfattn-layers $pointer_dist_decoder_selfattn_layers \
        --pointer-dist-decoder-selfattn-heads $pointer_dist_decoder_selfattn_heads \
        --pointer-dist-decoder-selfattn-avg $pointer_dist_decoder_selfattn_avg \
        --pointer-dist-decoder-selfattn-infer $pointer_dist_decoder_selfattn_infer \
        \
        --max-epoch $max_epoch \
        --arch transformer_tgt_pointer \
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
        --save-dir $MODEL_FOLDER \
        --tensorboard-logdir $MODEL_FOLDER

fi
