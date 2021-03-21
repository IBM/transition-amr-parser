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
config_data=config_files/config_data/config_data_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-base-top12.sh

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

arch=transformer_tgt_pointer_graphmp
tgt_graph_layers="0 1 2"
tgt_graph_heads=2
tgt_graph_mask="allprev_1in1out"

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

if [[ $tgt_graph_layers == "0 1 2 3 4 5" ]]; then
    grh_lay="all"
else
    grh_lay=""
    for n in $tgt_graph_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'tgt_graph_layers' input: $tgt_graph_layers" && exit 1
        grh_lay=$grh_lay$(( $n + 1 ))
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

grh_mask=-$tgt_graph_mask

if [[ $tgt_src_align_focus == "p0c1n0" ]]; then
    cam_focus=""    # default
elif [[ $tgt_src_align_focus == "p0c1n0 p0c0n*" ]]; then
    cam_focus=-abuf    # alignment and "buffer"
fi

# set the experiment directory name
expdir=exp_${data_tag}_act-pos-grh_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

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

# graph structure mask on the decoder self-attention
grh_tag=_grh-lay${grh_lay}-h${tgt_graph_heads}${grh_mask}

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

# combine different model configuration tags to the name
expdir=${expdir}${ptr_tag}${grh_tag}${cam_tag}${tis_tag}

# specific model directory name with a set random seed
MODEL_FOLDER=$ROOTDIR/$expdir/models_ep${max_epoch}_seed${seed}


###############################################################

##### decoding configuration
# model_epoch=_last
# # beam_size=1
# batch_size=128
