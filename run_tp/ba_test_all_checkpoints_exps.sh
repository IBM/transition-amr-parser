#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

exp_dirs=($rootdir/exp_o*)

epoch_last=120

for exp_dir in "${exp_dirs[@]}"; do

    echo -e "\n[Decoding for all checkpoints under experiments:]"
    echo "$exp_dir"

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do

        if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then

            # get the epoch number at the bottom of the sorted list, where
            # the sorting is used by 'run_tp/ba_test_all_checkpoints.sh'
            # bash array needs "( )" outside
            checkpoints_list=($(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r))
            epoch_bottom_sorted=$(basename ${checkpoints_list[-1]} | sed 's@checkpoint\(.*\)\.pt@\1@g')

            if [[ ! -f $checkpoints_folder/beam1/valid_checkpoint${epoch_bottom_sorted}.wiki.smatch ]]; then

                echo $checkpoints_folder

                jbsub_info=$(jbsub \
                         -cores 1+1 \
                         -mem 50g \
                         -q x86_6h \
                         -require v100 \
                         -name $0 \
                         -out $checkpoints_folder/logdec-%J.stdout \
                         -err $checkpoints_folder/logdec-%J.stderr \
                         /bin/bash run_tp/ba_test_all_checkpoints.sh $checkpoints_folder \
                         | grep 'is submitted to queue')

                echo $jbsub_info

                # debug: check if some checkpoints are unfinished
                # echo "UNFINISHED"

            else

                echo "[Following model finished decoding all checkpoints:]"
                echo "$checkpoints_folder"

                # if [[ ! -f $checkpoints_folder/checkpoint_wiki-smatch_top5-avg.pt ]]; then

                #     jbsub_info=$(jbsub \
                #              -cores 1 \
                #              -mem 20g \
                #              -q x86_6h \
                #              -name $0 \
                #              -out $checkpoints_folder/logrank-%J.stdout \
                #              -err $checkpoints_folder/logrank-%J.stderr \
                #              /bin/bash run_tp/bb_rank_all_checkpoints.sh $checkpoints_folder \
                #              | grep 'is submitted to queue')

                #     echo $jbsub_info

                # fi
            
            fi

        else

            echo "[Following model not trained till $epoch_last epochs:]"
            echo "$checkpoints_folder"

        fi

    done

    # to debug
    # break

done
