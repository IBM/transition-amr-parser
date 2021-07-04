#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

exp_dirs=($rootdir/exp_o*)
# exp_dirs=($rootdir/exp_o5_no-mw*)
# exp_dirs=($rootdir/exp_o3*)
# exp_dirs=($rootdir/exp_o5_no-mw* $rootdir/exp_o3*)    # both patterns
# exp_dirs=($rootdir/exp_o5?[!n]*)    # not exp_o5_no-mw*

epoch_last=120

for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Decoding for all checkpoints under experiments:]"
    echo "$exp_dir"

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do

        echo $checkpoints_folder

        if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then

            if [[ ! -f $checkpoints_folder/model-selection_stage3-done ]]; then

                echo "Do it here"

                jbsub_info=$(jbsub \
                             -cores 1+1 \
                             -mem 20g \
                             -q x86_6h \
                             -require v100 \
                             -name $0 \
                             -out $checkpoints_folder/logpost-%J.stdout \
                             -err $checkpoints_folder/logpost-%J.stderr \
                             /bin/bash run_tp/bd_testall-rank-avgtop-dec.sh $checkpoints_folder $epoch_last \
                             | grep 'is submitted to queue')

                echo $jbsub_info

            else

                echo "Done"

            fi

        else

            echo "[model not trained till $epoch_last epochs]"

        fi

    done

    # to debug
    # break

done
