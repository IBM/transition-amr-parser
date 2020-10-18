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

[ -z $1 ] && echo -e "\nRun by: bash $0 [config_model.sh]" && exit 1
set -o nounset

# load model configuration (including data)
config_model=$1
. $config_model

# decoding and testing setup
# model_epoch=_last
model_epoch=_wiki-smatch_top3-avg
beam_size=5
batch_size=128
use_pred_rules=0

# debug
model_epoch=106
beam_size=1

. run_tp/ad_test.sh $config_model dev
