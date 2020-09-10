#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

# exp_dirs=($rootdir/exp_o*)
exp_dirs=($rootdir/exp_o5_no-mw*)

epoch_last=120

for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Results for all model checkpoints under experiments:]"
    echo "$exp_dir"

    model_folders=($exp_dir/models*)

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
