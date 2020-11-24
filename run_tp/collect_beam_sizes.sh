#!/bin/bash

set -o errexit
set -o pipefail


# model_epoch=_last
# model_epoch=_wiki-smatch_best1
# model_epoch=_wiki-smatch_top3-avg
model_epoch=_wiki-smatch_top5-avg

## base model (6x6) on AMR 2.0
echo "base model (6x6) on AMR 2.0"
ls EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.wiki.smatch
echo
cat EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.wiki.smatch
echo

## small model (3x3) on AMR 2.0
echo "small model (6x6) on AMR 2.0"
ls EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed0/beam*/test_checkpoint${model_epoch}.wiki.smatch
echo
cat EXP/exp_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed0/beam*/test_checkpoint${model_epoch}.wiki.smatch
echo

model_epoch=_smatch_top5-avg

## base model (6x6) on AMR 1.0
echo "base model (6x6) on AMR 1.0"
ls EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.smatch
echo
cat EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.smatch
echo

## small model (3x3) on AMR 1.0
echo "small model (6x6) on AMR 1.0"
ls EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.smatch
echo
cat EXP/exp_amr1_graphmp-swaparc-ptrlast_o8.3_roberta-large-top24_3x3_act-pos-grh_vmask1_shiftpos1_ptr-lay3-h1_grh-lay12-h2-allprev_1in1out_cam-layall-h2-abuf/models_ep120_seed42/beam*/test_checkpoint${model_epoch}.smatch
echo

