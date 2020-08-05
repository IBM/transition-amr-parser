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
tgt_src_align_layer='all'
tgt_src_align_head=1
tgt_src_align_focus='p0n0'    # 'p0n1', 'p1n1' (alignment position, previous 1 position, next 1 position)

pointer_dist_decoder_selfattn_layers="0 1 2 3 4 5"
pointer_dist_decoder_selfattn_heads=2
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=5

seed=42
max_epoch=120

expdir=exp_${data_tag}_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}_cattnmask-layer${tgt_src_align_layer}-head${tgt_src_align_head}-focus${tgt_src_align_focus}_ptr-layerall-head${pointer_dist_decoder_selfattn_heads}    # action-pointer

MODEL_FOLDER=$ROOTDIR/$expdir/models_ep${max_epoch}_seed${seed}


###############################################################

##### decoding configuration
# model_epoch=_last
# # beam_size=1
# batch_size=128

