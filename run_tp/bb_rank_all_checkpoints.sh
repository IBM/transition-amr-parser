#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\ntest_all_checkpoints.sh <checkpoints_folder>\n" && \
    exit 1
set -o nounset


# model ranking and linking
python run_tp/bb_rank_model.py \
       --checkpoints $checkpoints_folder \
       --link_best 5 \
       --score wiki.smatch


# average checkpoint
if [[ -f $checkpoints_folder/checkpoint_wiki-smatch_best3.pt ]]; then

    python fairseq/scripts/average_checkpoints.py \
            --input \
                $checkpoints_folder/checkpoint_wiki-smatch_best1.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best2.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best3.pt \
            --output $checkpoints_folder/checkpoint_wiki-smatch_top3-avg.pt

fi


if [[ -f $checkpoints_folder/checkpoint_wiki-smatch_best5.pt ]]; then

    python fairseq/scripts/average_checkpoints.py \
            --input \
                $checkpoints_folder/checkpoint_wiki-smatch_best1.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best2.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best3.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best4.pt \
                $checkpoints_folder/checkpoint_wiki-smatch_best5.pt \
            --output $checkpoints_folder/checkpoint_wiki-smatch_top5-avg.pt

fi

touch $checkpoints_folder/model-selection_stage2-done


