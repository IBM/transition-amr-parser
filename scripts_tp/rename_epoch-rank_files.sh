#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP
rootdir=EXP

exp_dirs=($rootdir/exp_o*)


for exp_dir in "${exp_dirs[@]}"; do

    echo "$exp_dir"

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do
    
        if [[ -f $checkpoints_folder/epoch_wiki-smatch_ranks.txt ]]; then
            mv $checkpoints_folder/epoch_wiki-smatch_ranks.txt $checkpoints_folder/epoch_wiki-smatch_ranks.txt.predrules
        fi
        
    done

done


