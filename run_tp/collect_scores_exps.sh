#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

# rootdir=/dccstor/jzhou1/work/EXP
rootdir=EXP

# exp_dirs=($rootdir/exp_o*)
# exp_dirs=($rootdir/exp_o3*)
# exp_dirs=($rootdir/exp_o5?[!n]*)
# exp_dirs=($rootdir/exp_o5_no-mw*)
# epoch_last=120

# seed=0
seed=""    # check all seeds

# exp_dirs=($rootdir/exp_amr1*)
# epoch_last=120

# exp_dirs=($rootdir/exp_depfix_o5*)
# epoch_last=150

exp_dirs=($rootdir/exp_o5*)
epoch_last=120


for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Results for all model checkpoints under experiments:]"
    echo "$exp_dir"

    # model_folders=($exp_dir/models*)

    if [[ $seed == "" ]]; then
        model_folders=($exp_dir/models*)
    else
        model_folders=($exp_dir/models_ep${epoch_last}_seed${seed})
    fi

    for checkpoints_folder in "${model_folders[@]}"; do

        echo $checkpoints_folder

        if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then

            bash run_tp/collect_scores.sh $checkpoints_folder

        else

            echo "[model not trained till $epoch_last epochs]"

        fi

    done

    # to debug
    # break

done
