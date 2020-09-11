#!/bin/bash

# remove all checkpoints that are with name "checkpoints[0-9]*",
# but are not linked by the best models via model selection

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

# exp_dirs=($rootdir/exp_o*)
# exp_dirs=($rootdir/exp_o5_no-mw*)
exp_dirs=($rootdir/exp_o3*)
# exp_dirs=($rootdir/exp_o5_no-mw* $rootdir/exp_o3*)    # both patterns
# exp_dirs=($rootdir/exp_o5?[!n]*)    # o5 but not exp_o5_no-mw*

epoch_last=120

for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Removing unused checkpoints under experiments:]"
    echo "$exp_dir"

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do
    
        echo $checkpoints_folder
        
        if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then
        
            if [[ -f $checkpoints_folder/model-selection_stage3-done ]]; then
            
                echo "removing unlinked checkpoints"

                python run_tp/be_rm_checkpoints.py $checkpoints_folder

            else

                echo "Model selection and best model linking NOT done"

            fi
        
        else
        
            # after removing the unlinked checkpoints, the checkpoint${epoch_last}.pt could disappear too
            # so it will fall under this category
            if [[ ! -f $checkpoints_folder/model-selection_stage3-done ]]; then
        
                echo "[model not trained till $epoch_last epochs] removing all checkpoints"
            
                for dfile in $(find $checkpoints_folder -type f -iname 'checkpoint[0-9]*.pt' | sort -n); do
                    echo $dfile
                    # rm $dfile    # NOTE be very careful of this; do not make mistake to remove linked checkpoints
                done

            fi
        
        fi

    done

    # to debug
    # break

done

