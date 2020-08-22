#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### root folder to store everything
ROOTDIR=/dccstor/jzhou1/work/EXP

##############################################################

##### load data config
config_data=config_data_o3_roberta-large-top24.sh
# config_data=config_data_o3_roberta-base-top12.sh

data_tag="$(basename $config_data | sed 's@config_data_\(.*\)\.sh@\1@g')"


dir=$(dirname $0)
. $dir/$config_data   # we should always call from one level up
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $EMB_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

###############################################################

##### model configuration
shift_pointer_value=0
tgt_vocab_masks=0
share_decoder_embed=0

apply_tgt_src_align=1
tgt_src_align_layers="0 1 2 3 4 5"
tgt_src_align_heads=2
tgt_src_align_focus="p0c1n0 p0c1n0"
focus_name='aa'
# previous version: 'p0n1', 'p1n1' (alignment position, previous 1 position, next 1 position)
# current version: 'p0c1n1', 'p1c1n1', 'p*c1n0', 'p0c0n*', etc.
#                  'p' - previous (prior to alignment), a number or '*' for all previous src tokens
#                  'c' - current (alignment position, 1 for each tgt token), either 0 or 1
#                  'n' - next (post alignment), a number or '*' for all the remaining src tokens

pointer_dist_decoder_selfattn_layers="5"
pointer_dist_decoder_selfattn_heads=1
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=5

apply_tgt_input_src=0
tgt_input_src_emb=top
tgt_input_src_backprop=1
tgt_input_src_combine=cat

seed=${seed:-42}
max_epoch=120

cam_lay="all"

if [[ $pointer_dist_decoder_selfattn_layers == "0 1 2 3 4 5" ]]; then
    lay="all"
elif [[ $pointer_dist_decoder_selfattn_layers == "3 4 5" ]]; then
    lay="456"
elif [[ $pointer_dist_decoder_selfattn_layers == "4 5" ]]; then
    lay="56"
elif [[ $pointer_dist_decoder_selfattn_layers == "5" ]]; then
    lay="6"
else
    echo "Invalid 'pointer_dist_decoder_selfattn_layers' input: $pointer_dist_decoder_selfattn_layers" && exit 0
fi


expdir=exp_${data_tag}_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

if [[ $apply_tgt_src_align == 1 ]]; then
    expdir=${expdir}_cam-layer${cam_lay}-head${tgt_src_align_heads}-focus${focus_name}
fi

expdir=${expdir}_ptr-layer${lay}-head${pointer_dist_decoder_selfattn_heads}    # action-pointer


if [[ $pointer_dist_decoder_selfattn_heads != 1 ]]; then
    if [[ $pointer_dist_decoder_selfattn_avg == 1 ]]; then
        expdir=${expdir}-avg
    elif [[ $pointer_dist_decoder_selfattn_avg == "-1" ]]; then
        expdir=${expdir}-apd
    fi
fi

if [[ $apply_tgt_input_src == 1 ]]; then
    expdir=${expdir}_tis-emb${tgt_input_src_emb}-com${tgt_input_src_combine}-bp${tgt_input_src_backprop}
fi

MODEL_FOLDER=$ROOTDIR/$expdir/models_ep${max_epoch}_seed${seed}


###############################################################

##### decoding configuration
# model_epoch=_last
# # beam_size=1
# batch_size=128

