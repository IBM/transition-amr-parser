#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

# exp_dirs=($rootdir/exp_o*)
# exp_dirs=($rootdir/exp_o5_no-mw*)
# epoch_last=120

# seed=0
seed=""    # check all seeds

exp_dirs=($rootdir/exp_depfix_o5*)
epoch_last=150

for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Results for all model checkpoints under experiments:]"
    echo "$exp_dir"

    if [[ $seed == "" ]]; then
        model_folders=($exp_dir/models*)
    else
        model_folders=($exp_dir/models_ep${epoch_last}_seed${seed})
    fi

    for checkpoints_folder in "${model_folders[@]}"; do
    
        echo $checkpoints_folder
        
        if [[ -f $checkpoints_folder/epoch_wiki-smatch_ranks.txt ]]; then
        
            head -6 $checkpoints_folder/epoch_wiki-smatch_ranks.txt
        
        else
        
            echo "[epochs not tested and ranked]"
        
        fi

    done

    # to debug
    # break

done
