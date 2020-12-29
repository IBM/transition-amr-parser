#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ ! -f "$1" ] && "Missing $1" && exit 1
config=$1
# random seed
[ -z "$2" ] && echo -e "$HELP" && exit 1
seed=$2

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# ##### script specific config
# if [ -z ${max_epoch+x} ]; then
#     max_epoch=120
# fi
# eval_init_epoch=${eval_init_epoch:-81}

##### TRAINING
# rm -Rf $MODEL_FOLDER

# if [ -f $MODEL_FOLDER/checkpoint_last.pt ]; then

#     echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt already exists --- do nothing."

if [ -f ${MODEL_FOLDER}-seed${seed}/checkpoint_last.pt ] && [ -f ${MODEL_FOLDER}-seed${seed}/checkpoint${MAX_EPOCH}.pt ]; then

    echo "Model checkpoint ${MODEL_FOLDER}-seed${seed}/checkpoint_last.pt && ${MODEL_FOLDER}-seed${seed}/checkpoint${MAX_EPOCH}.pt already exist --- do nothing."

else

    # if [[ $arch == "transformer_tgt_pointer" ]]; then
    if [[ $arch != *"graph"* ]]; then

        # python -m ipdb fairseq_ext/train.py \
        python fairseq_ext/train.py \
            $DATA_FOLDER \
            --emb-dir $EMB_FOLDER \
            --user-dir ../fairseq_ext \
            --task $TASK \
            --append-eos-to-target 0 \
            --collate-tgt-states 1 \
            --shift-pointer-value $shift_pointer_value \
            --apply-tgt-vocab-masks $tgt_vocab_masks \
            --share-decoder-input-output-embed $share_decoder_embed \
            --tgt-factored-emb-out $tgt_factored_emb_out \
            \
            --apply-tgt-src-align $apply_tgt_src_align \
            --tgt-src-align-layers $tgt_src_align_layers \
            --tgt-src-align-heads $tgt_src_align_heads \
            --tgt-src-align-focus $tgt_src_align_focus \
            \
            --pointer-dist-decoder-selfattn-layers $pointer_dist_decoder_selfattn_layers \
            --pointer-dist-decoder-selfattn-heads $pointer_dist_decoder_selfattn_heads \
            --pointer-dist-decoder-selfattn-avg $pointer_dist_decoder_selfattn_avg \
            --pointer-dist-decoder-selfattn-infer $pointer_dist_decoder_selfattn_infer \
            \
            --apply-tgt-actnode-masks $apply_tgt_actnode_masks \
            \
            --apply-tgt-input-src $apply_tgt_input_src \
            --tgt-input-src-emb $tgt_input_src_emb \
            --tgt-input-src-backprop $tgt_input_src_backprop \
            --tgt-input-src-combine $tgt_input_src_combine \
            \
            --max-epoch $MAX_EPOCH \
            --arch $arch \
            --optimizer adam \
            --adam-betas '(0.9,0.98)' \
            --clip-norm 0.0 \
            --lr-scheduler inverse_sqrt \
            --warmup-init-lr 1e-07 \
            --warmup-updates $warmup \
            --pretrained-embed-dim $PRETRAINED_EMBED_DIM \
            --lr $lr \
            --min-lr 1e-09 \
            --dropout $dropout \
            --weight-decay 0.0 \
            --criterion label_smoothed_cross_entropy_pointer \
            --label-smoothing 0.01 \
            --loss-coef 1 \
            --keep-last-epochs $(( $MAX_EPOCH - $EVAL_INIT_EPOCH + 1 )) \
            --max-tokens $max_tokens \
            --log-format json \
            --seed $seed \
            --save-dir ${MODEL_FOLDER}-seed${seed} \
            --tensorboard-logdir ${MODEL_FOLDER}-seed${seed}

    else

        # with graph structure

        python fairseq_ext/train.py \
            $DATA_FOLDER \
            --emb-dir $EMB_FOLDER \
            --user-dir ../fairseq_ext \
            --task $TASK \
            --append-eos-to-target 0 \
            --collate-tgt-states 1 \
            --shift-pointer-value $shift_pointer_value \
            --apply-tgt-vocab-masks $tgt_vocab_masks \
            --share-decoder-input-output-embed $share_decoder_embed \
            \
            --apply-tgt-src-align $apply_tgt_src_align \
            --tgt-src-align-layers $tgt_src_align_layers \
            --tgt-src-align-heads $tgt_src_align_heads \
            --tgt-src-align-focus $tgt_src_align_focus \
            \
            --pointer-dist-decoder-selfattn-layers $pointer_dist_decoder_selfattn_layers \
            --pointer-dist-decoder-selfattn-heads $pointer_dist_decoder_selfattn_heads \
            --pointer-dist-decoder-selfattn-avg $pointer_dist_decoder_selfattn_avg \
            --pointer-dist-decoder-selfattn-infer $pointer_dist_decoder_selfattn_infer \
            \
            --apply-tgt-actnode-masks $apply_tgt_actnode_masks \
            \
            --apply-tgt-input-src $apply_tgt_input_src \
            --tgt-input-src-emb $tgt_input_src_emb \
            --tgt-input-src-backprop $tgt_input_src_backprop \
            --tgt-input-src-combine $tgt_input_src_combine \
            \
            --tgt-graph-layers $tgt_graph_layers \
            --tgt-graph-heads $tgt_graph_heads \
            --tgt-graph-mask $tgt_graph_mask \
            \
            --max-epoch $MAX_EPOCH \
            --arch $arch \
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
            --keep-last-epochs $(( $MAX_EPOCH - $EVAL_INIT_EPOCH + 1 )) \
            --max-tokens 3584 \
            --log-format json \
            --seed $seed \
            --save-dir ${MODEL_FOLDER}-seed${seed} \
            --tensorboard-logdir ${MODEL_FOLDER}-seed${seed}
    
    fi

    # Mark as finished
    touch ${MODEL_FOLDER}-seed${seed}/.done

fi
