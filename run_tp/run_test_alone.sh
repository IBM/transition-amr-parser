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

# config_model=EXP/exp_debug6/config_model_debug.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_bart-enc-fix/models_ep120_seed42_lr0.00005-mt2048-wm4000-dp0.3/config_model_apt-bart_*.sh
config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_bart-enc-fix_bart-emb-fix/models_ep120_seed42_lr0.00005-mt2048-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_bart-init0/models_ep120_seed42_lr0.00005-mt2048-wm4000-dp0.3/config_model_apt-bart_data_o5_ptr-lay6-h1_cam-layall-h2-abuf_bart-init0.sh
config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0/models_ep120_seed42_lr0.0001-mt3584-wm4000-dp0.2/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha1/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha1_bart-init0/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha1_bart-enc-fix/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh

# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-dec-emb-in/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-enc0/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh

# config_model=EXP/exp_o8.3_bart-large_act-pos_vmask1_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh

# config_model=EXP/exp_o8.3_roberta-base-top12-wp_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_enc-pool-top/models_ep120_seed42_lr0.0005-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh

# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_enc-roberta-base/models_ep120_seed42_lr0.0001-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init0_enc-roberta-base/models_ep120_seed42_lr0.0001-mt3584-wm4000-dp0.3/config_model_apt-bart_*.sh
# config_model=EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/config_model_action-pointer-graphmp_data_o5_ptr-lay6-h1_grh-h2-allprev_1in1out_cam-layall-h2-abuf.sh

config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf/models_ep120_seed42_lr0.0001-mt3584-wm4000-dp0.2/config_model_apt-bart_*.sh
config_model=EXP/exp_o8.3_bart-base_act-pos_vmask1_shiftpos1_ptr-lay6-h1_cam-layall-h2-abuf_dec-sep-emb-sha1_bart-dec-emb-comp-pred/models_ep120_seed42_lr0.0001-mt3584-wm4000-dp0.2/config_*.sh

config_model=EXP/exp_o10_bart-large_act-pos_vmask1_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb/models_ep120_seed42_lr0.00005-mt3584-wm4000-dp0.2/config_model_apt-bart_*.sh
# config_model=EXP/exp_o9.0_bart-large_act-pos_vmask1_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb/models_ep120_seed42_lr0.0001-mt2048x4-wm4000-dp0.2/config_*.sh

. $config_model

# decoding and testing setup
# model_epoch=_last
# model_epoch=_wiki-smatch_top3-avg
# model_epoch=_wiki-smatch_top5-avg
# beam_size=5
batch_size=128
use_pred_rules=0

# debug
model_epoch=40
beam_size=1

# . run_tp/ad_test.sh $config_model test
. run_tp/ad_test.sh $config_model dev
