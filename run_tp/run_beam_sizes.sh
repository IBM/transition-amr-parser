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
. set_environment.sh

#[ -z $1 ] && echo -e "\nRun by: bash $0 [config_model.sh]" && exit 1
#set -o nounset

# load model configuration (including data)
#config_model=$1

## base model (6x6) on AMR 2.0
config_model=EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/config_model_action-pointer-graphmp_data_o5_ptr-lay6-h1_grh-h2-allprev_1in1out_cam-layall-h2-abuf.sh
## small model (3x3) on AMR 2.0
config_model=EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed0/config_model_*
## small model (3x3) on AMR 1.0
config_model=EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/config_model_*
## base model (6x6) on AMR 1.0
config_model=EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/config_model_*

. $config_model

# decoding and testing setup
# model_epoch=_last
# model_epoch=_wiki-smatch_best1
# model_epoch=_wiki-smatch_top3-avg
model_epoch=_wiki-smatch_top5-avg

# for AMR 1.0 no wiki
if [[ $config_model == *amr1* ]]; then
    model_epoch=_smatch_top5-avg
fi

beam_size=2
batch_size=128
use_pred_rules=0

# debug
# model_epoch=106
# beam_size=1

num_runs=3
beams=(2 3 4)
gpus=(0 1 2 3)

#num_runs=4
#beams=(6 7 8 9)
#gpus=(0 1 2 3)

for (( i=0; i<$num_runs; i++ ))
do
    beam_size=${beams[i]}
    gpu=${gpus[i]}
    # will all output to screen
    (CUDA_VISIBLE_DEVICES=$gpu . run_tp/ad_test.sh $config_model test) &

done


