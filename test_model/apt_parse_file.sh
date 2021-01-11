#!/bin/bash

set -e
set -o pipefail
set -o nounset

in_checkpoint=EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/checkpoint_wiki-smatch_top5-avg.pt
input_file=test_model/gigaword_ref.txt


amr-parse \
    --in-checkpoint $in_checkpoint \
    --in-tokenized-sentences $input_file \
    --out-amr file.amr
