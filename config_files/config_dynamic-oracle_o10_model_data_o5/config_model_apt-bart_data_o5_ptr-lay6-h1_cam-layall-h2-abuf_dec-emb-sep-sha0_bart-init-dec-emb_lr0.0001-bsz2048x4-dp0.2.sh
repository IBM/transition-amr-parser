#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### root folder to store everything
. set_exps.sh    # general setup for experiments management (save dir, etc.)

if [ -z ${ROOTDIR+x} ]; then
    ROOTDIR=EXP
fi

##############################################################

##### load data config
config_data=config_files/config_data/config_data_dynamic-oracle_o10_bart-base.sh

data_tag="$(basename $config_data | sed 's@config_data_\(.*\)\.sh@\1@g')"


dir=$(dirname $0)
. $config_data   # $config_data should include its path
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $EMB_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

###############################################################

##### model configuration
shift_pointer_value=1
apply_tgt_actnode_masks=0
tgt_vocab_masks=1
share_decoder_embed=0

arch=transformer_tgt_pointer_bart_base

initialize_with_bart=1
initialize_with_bart_enc=1
initialize_with_bart_dec=1
bart_encoder_backprop=1
bart_emb_backprop=1
bart_emb_decoder=0
bart_emb_decoder_input=0
bart_emb_init_composition=1

pointer_dist_decoder_selfattn_layers="5"
pointer_dist_decoder_selfattn_heads=1
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=5

apply_tgt_src_align=1
tgt_src_align_layers="0 1 2 3 4 5"
tgt_src_align_heads=2
tgt_src_align_focus="p0c1n0 p0c0n*"
# previous version: 'p0n1', 'p1n1' (alignment position, previous 1 position, next 1 position)
# current version: 'p0c1n1', 'p1c1n1', 'p*c1n0', 'p0c0n*', etc.
#                  'p' - previous (prior to alignment), a number or '*' for all previous src tokens
#                  'c' - current (alignment position, 1 for each tgt token), either 0 or 1
#                  'n' - next (post alignment), a number or '*' for all the remaining src tokens

apply_tgt_input_src=0
tgt_input_src_emb=top
tgt_input_src_backprop=1
tgt_input_src_combine="add"

seed=${seed:-42}
max_epoch=120
eval_init_epoch=81
# max_epoch=5
# eval_init_epoch=1

lr=0.0001
max_tokens=2048
update_freq=4
warmup=4000
dropout=0.2


##### set the experiment dir name based on model configurations

if [[ $pointer_dist_decoder_selfattn_layers == "0 1 2 3 4 5" ]]; then
    lay="all"
else
    lay=""
    for n in $pointer_dist_decoder_selfattn_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'pointer_dist_decoder_selfattn_layers' input: $pointer_dist_decoder_selfattn_layers" && exit 1
        lay=$lay$(( $n + 1 ))
    done
fi


if [[ $tgt_src_align_layers == "0 1 2 3 4 5" ]]; then
    cam_lay="all"
else
    cam_lay=""
    for n in $tgt_src_align_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'tgt_src_align_layers' input: $tgt_src_align_layers" && exit 1
        cam_lay=$cam_lay$(( $n + 1 ))
    done
fi


if [[ $tgt_src_align_focus == "p0c1n0" ]]; then
    cam_focus=""    # default
elif [[ $tgt_src_align_focus == "p0c1n0 p0c0n*" ]]; then
    cam_focus=-abuf    # alignment and "buffer"
fi

# set the experiment directory name
expdir=exp_${data_tag}_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

# pointer distribution
ptr_tag=_ptr-lay${lay}-h${pointer_dist_decoder_selfattn_heads}    # action-pointer

if [[ $pointer_dist_decoder_selfattn_avg == 1 ]]; then
    ptr_tag=${ptr_tag}-avg
elif [[ $pointer_dist_decoder_selfattn_avg == "-1" ]]; then
    ptr_tag=${ptr_tag}-apd
fi

if [[ $apply_tgt_actnode_masks == 1 ]]; then
    ptr_tag=${ptr_tag}-pmask1
fi

# cross-attention alignment
if [[ $apply_tgt_src_align == 1 ]]; then
    cam_tag=_cam-lay${cam_lay}-h${tgt_src_align_heads}${cam_focus}
else
    cam_tag=""
fi

# target input augmentation
if [[ $apply_tgt_input_src == 1 ]]; then
    tis_tag=_tis-emb${tgt_input_src_emb}-com${tgt_input_src_combine}-bp${tgt_input_src_backprop}
else
    tis_tag=""
fi

# initialize with bart
if [[ $initialize_with_bart == 0 ]]; then
    init_tag=_bart-init${initialize_with_bart}
else
    if [[ $initialize_with_bart_enc == 0 ]]; then
        [[ $initialize_with_bart_dec == 0 ]] && echo "initialize_with_bart_dec should be 1 here" && exit 1
        init_tag=_bart-init-enc0
    fi
    if [[ $initialize_with_bart_dec == 0 ]]; then
        [[ $initialize_with_bart_enc == 0 ]] && echo "initialize_with_bart_enc should be 1 here" && exit 1
        init_tag=_bart-init-dec0
    fi
    if [[ $initialize_with_bart_enc == 1 ]] && [[ $initialize_with_bart_dec == 1 ]]; then
        init_tag=""
    fi
fi

# fix bart encoder
if [[ $bart_encoder_backprop == 0 ]]; then
    [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
    enc_fix_tag=_bart-enc-fix
else
    enc_fix_tag=""
fi

# fix bart embedding
if [[ $bart_emb_backprop == 0 ]]; then
    [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
    emb_fix_tag=_bart-emb-fix
else
    emb_fix_tag=""
fi

# separate decoder embedding
if [[ $bart_emb_decoder == 0 ]]; then
    dec_emb_tag=_dec-sep-emb-sha${share_decoder_embed}
else
    dec_emb_tag=""
fi
# bart decoder input embedding
if [[ $bart_emb_decoder_input == 0 ]]; then
    [[ $bart_emb_decoder == 1 ]] && echo "bart_emb_decoder should be 0" && exit 1
    dec_emb_in_tag=""
else
    if [[ $bart_emb_decoder == 1 ]]; then
        dec_emb_in_tag=""
    else
        # decoder input BART embeddings, output customized embeddings
        dec_emb_in_tag="_bart-dec-emb-in"
    fi
fi
# initialize target embedding with compositional sub-token embeddings
if [[ $bart_emb_init_composition == 1 ]]; then
    dec_emb_init_tag="_bart-init-dec-emb"
else
    dec_emb_init_tag=""
fi

# combine different model configuration tags to the name
expdir=${expdir}${ptr_tag}${cam_tag}${tis_tag}${dec_emb_tag}${dec_emb_in_tag}${dec_emb_init_tag}${init_tag}${enc_fix_tag}${emb_fix_tag}


# specific model directory name with a set random seed
optim_tag=_lr${lr}-mt${max_tokens}x${update_freq}-wm${warmup}-dp${dropout}
MODEL_FOLDER=$ROOTDIR/$expdir/models_ep${max_epoch}_seed${seed}${optim_tag}



###############################################################

##### decoding configuration
# model_epoch=_last
# # beam_size=1
# batch_size=128
