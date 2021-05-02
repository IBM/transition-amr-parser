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
        . $config    # $config_model should always include its path
    fi
    # NOTE: when the first configuration argument is not provided, this script must
    #       be called from other scripts
fi


##### script specific config
if [ -z ${max_epoch+x} ]; then
    max_epoch=120
fi
eval_init_epoch=${eval_init_epoch:-81}
seed=${seed:-42}
# max_epoch=120
# seed=42

# $TASK is defined in the data configuration file
# $arch is defined in the model configuration file
TASK=${TASK:-amr_action_pointer}
arch=${arch:-transformer_tgt_pointer}

apply_tgt_input_src=${apply_tgt_input_src:-0}
apply_tgt_actnode_masks=${apply_tgt_actnode_masks:-0}

tgt_input_src_emb=${tgt_input_src_emb:-top}
tgt_input_src_backprop=${tgt_input_src_backprop:-1}
tgt_input_src_combine=${tgt_input_src_combine:-cat}

if [[ $arch == "transformer_tgt_pointer_graphmp" ]]; then
    tgt_graph_mask=${tgt_graph_mask:-1prev}
fi

tgt_graph_mask=${tgt_graph_mask:-e1c1p1}

tgt_factored_emb_out=${tgt_factored_emb_out:-0}

initialize_with_bart=${initialize_with_bart:-1}
initialize_with_bart_enc=${initialize_with_bart_enc:-1}
initialize_with_bart_dec=${initialize_with_bart_dec:-1}
bart_encoder_backprop=${bart_encoder_backprop:-1}
bart_emb_backprop=${bart_emb_backprop:-1}
bart_emb_decoder=${bart_emb_decoder:-1}
bart_emb_decoder_input=${bart_emb_decoder_input:-1}
bart_emb_init_composition=${bart_emb_init_composition:-0}
bart_emb_composition_pred=${bart_emb_composition_pred:-0}

src_roberta_emb=${src_roberta_emb:-0}
src_fix_emb_use=$src_roberta_emb
src_pool_wp2w=${src_pool_wp2w:-top}
src_avg_layers=${src_avg_layers:-""}
src_roberta_enc=${src_roberta_enc:-0}

# for apt-bart shared vocabulary
node_freq_min=${NODE_FREQ_MIN:-5}

lr=${lr:-0.0005}
max_tokens=${max_tokens:-3584}
update_freq=${update_freq:-1}
warmup=${warmup:-4000}
dropout=${dropout:-0.3}
clip_norm=${clip_norm:-0.0}

weight_decay=${weight_decay:-0.0}
loss_coef=${loss_coef:-1}

##### TRAINING
# rm -Rf $MODEL_FOLDER

# if [ -f $MODEL_FOLDER/checkpoint_last.pt ]; then

#     echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt already exists --- do nothing."

if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

    # if [[ $arch == "transformer_tgt_pointer" ]]; then
    if [[ $arch != *"graph"* ]]; then

    if [[ $arch != *"bartsv"* ]]; then
    # apt-bart, with separate src and tgt vocabulary

    # python -m ipdb fairseq_ext/train.py \
    python fairseq_ext/train.py \
        $DATA_FOLDER \
        --emb-dir $EMB_FOLDER \
        --user-dir fairseq_ext \
        --task $TASK \
        --append-eos-to-target 0 \
        --collate-tgt-states 1 \
        --src-fix-emb-use $src_fix_emb_use \
        --shift-pointer-value $shift_pointer_value \
        --apply-tgt-vocab-masks $tgt_vocab_masks \
        --share-decoder-input-output-embed $share_decoder_embed \
        --tgt-factored-emb-out $tgt_factored_emb_out \
        \
        --initialize-with-bart $initialize_with_bart \
        --initialize-with-bart-enc $initialize_with_bart_enc \
        --initialize-with-bart-dec $initialize_with_bart_dec \
        --bart-encoder-backprop $bart_encoder_backprop \
        --bart-emb-backprop $bart_emb_backprop \
        --bart-emb-decoder $bart_emb_decoder \
        --bart-emb-decoder-input $bart_emb_decoder_input \
        --bart-emb-init-composition $bart_emb_init_composition \
        --bart-emb-composition-pred $bart_emb_composition_pred \
        \
        --src-roberta-emb $src_roberta_emb \
        --src-pool-wp2w $src_pool_wp2w \
        --src-avg-layers $src_avg_layers \
        --src-roberta-enc $src_roberta_enc \
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
        --max-epoch $max_epoch \
        --arch $arch \
        --optimizer adam \
        --adam-betas '(0.9,0.98)' \
        --clip-norm $clip_norm \
        --lr-scheduler fixed \
        --warmup-updates $warmup \
        --pretrained-embed-dim $PRETRAINED_EMBED_DIM \
        --lr $lr \
        --min-lr 1e-09 \
        --dropout $dropout \
        --weight-decay $weight_decay \
        --criterion label_smoothed_cross_entropy_pointer \
        --label-smoothing 0.01 \
        --loss-coef $loss_coef \
        --keep-last-epochs $(( $max_epoch - $eval_init_epoch + 1 )) \
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --log-format json \
        --seed $seed \
        --save-dir $MODEL_FOLDER \
        --tensorboard-logdir $MODEL_FOLDER

    else
    # apt-bart with shared and mixed src and tgt vocabulary

    # python -m ipdb fairseq_ext/train.py \
    python fairseq_ext/train.py \
        $DATA_FOLDER \
        --emb-dir $EMB_FOLDER \
        --user-dir fairseq_ext \
        --task $TASK \
        --node-freq-min $node_freq_min \
        --append-eos-to-target 0 \
        --collate-tgt-states 1 \
        --src-fix-emb-use $src_fix_emb_use \
        --shift-pointer-value $shift_pointer_value \
        --apply-tgt-vocab-masks $tgt_vocab_masks \
        --share-decoder-input-output-embed $share_decoder_embed \
        --share-all-embeddings ${share_all_embeddings:-1} \
        --tgt-factored-emb-out $tgt_factored_emb_out \
        \
        --initialize-with-bart $initialize_with_bart \
        --initialize-with-bart-enc $initialize_with_bart_enc \
        --initialize-with-bart-dec $initialize_with_bart_dec \
        --bart-encoder-backprop $bart_encoder_backprop \
        --bart-emb-backprop $bart_emb_backprop \
        --bart-emb-init-composition $bart_emb_init_composition \
        \
        --src-roberta-emb $src_roberta_emb \
        --src-pool-wp2w $src_pool_wp2w \
        --src-avg-layers $src_avg_layers \
        --src-roberta-enc $src_roberta_enc \
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
        --max-epoch $max_epoch \
        --arch $arch \
        --optimizer adam \
        --adam-betas '(0.9,0.98)' \
        --clip-norm $clip_norm \
        --lr-scheduler fixed \
        --warmup-updates $warmup \
        --pretrained-embed-dim $PRETRAINED_EMBED_DIM \
        --lr $lr \
        --min-lr 1e-09 \
        --dropout $dropout \
        --weight-decay $weight_decay \
        --criterion label_smoothed_cross_entropy_pointer \
        --label-smoothing 0.01 \
        --loss-coef $loss_coef \
        --keep-last-epochs $(( $max_epoch - $eval_init_epoch + 1 )) \
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --log-format json \
        --seed $seed \
        --save-dir $MODEL_FOLDER \
        --tensorboard-logdir $MODEL_FOLDER

    fi

    else

    # with graph structure

    python fairseq_ext/train.py \
        $DATA_FOLDER \
        --emb-dir $EMB_FOLDER \
        --user-dir fairseq_ext \
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
        --max-epoch $max_epoch \
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
        --keep-last-epochs $(( $max_epoch - $eval_init_epoch + 1 )) \
        --max-tokens 3584 \
        --log-format json \
        --seed $seed \
        --save-dir $MODEL_FOLDER \
        --tensorboard-logdir $MODEL_FOLDER

    fi

fi
