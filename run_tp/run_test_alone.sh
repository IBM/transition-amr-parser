#!/bin/bash

# for a single test of a model score
# for different variations, change
# a) the model specific configuration
# b) decoding setup (see below)
# to run:
# bash run_tp/ad_test.sh

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

#[ -z $1 ] && echo -e "\nRun by: bash $0 [config_model.sh]" && exit 1
#set -o nounset

# load model configuration (including data)
#config_model=$1

config_model=EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/config_model_action-pointer-graphmp_data_o5_ptr-lay6-h1_grh-h2-allprev_1in1out_cam-layall-h2-abuf.sh
. $config_model

# decoding and testing setup
# model_epoch=_last
# model_epoch=_wiki-smatch_top3-avg
model_epoch=_wiki-smatch_top5-avg
beam_size=1
batch_size=128
use_pred_rules=0

# debug
# model_epoch=106
# beam_size=1

. run_tp/ad_test.sh $config_model test
