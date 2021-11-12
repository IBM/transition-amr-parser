#!/bin/bash
set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
HELP="$0 <checkpoint> <tokenized sentences file>"
[ "$#" -lt 2 ] && echo "$HELP" && exit 1
input_file=$1
output_amr=$2
set -o nounset

amr-parse \
    --in-checkpoint DATA/AMR3.0/models/exp_cofill_o10_act-states_bart.large/_act-pos_vmask1_shiftpos1_ptr-lay12-h1_cam-layall-h2-abuf_dec-sep-emb-sha0_bart-init-dec-emb__fp16-_lr0.0001-mt2048x4-wm4000-dp0.2/ep120-seed42/checkpoint_wiki.smatch_top5-avg.pt \
    --in-machine-config DATA/AMR3.0/oracles/cofill_o10_act-states/machine_config.json \
    --in-tokenized-sentences $input_file \
    --out-amr $output_amr \
    --roberta-cache-path /dccstor/ysuklee1/RoBERTa/bart.large \
    --batch-size 128 \
    --roberta-batch-size 1
