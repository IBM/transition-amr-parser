#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\ntest_all_checkpoints.sh <checkpoints_folder>\n" && \
    exit 1

epoch_last=$2

set -o nounset


epoch_last=${epoch_last:-120}


if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then

    echo -e "\n$checkpoints_folder"

    echo "---------- stage 1: decoding all checkpoints ----------"

    if [[ ! -f $checkpoints_folder/model-selection_stage1-done ]]; then

        # jbsub_info=$(jbsub \
        #          -cores 1+1 \
        #          -mem 50g \
        #          -q x86_6h \
        #          -require v100 \
        #          -name $0 \
        #          -out $checkpoints_folder/logdec-%J.stdout \
        #          -err $checkpoints_folder/logdec-%J.stderr \
        #          /bin/bash run_tp/ba_test_all_checkpoints.sh $checkpoints_folder \
        #          | grep 'is submitted to queue')

        # echo $jbsub_info

        bash run_tp/ba_test_all_checkpoints.sh $checkpoints_folder

    else

        echo "Done"

    fi

    echo "---------- stage 2: rank all checkpoints and get averaged models ----------"

    if [[ ! -f $checkpoints_folder/model-selection_stage2-done ]]; then

        # jbsub_info=$(jbsub \
        #          -cores 1 \
        #          -mem 20g \
        #          -q x86_6h \
        #          -name $0 \
        #          -out $checkpoints_folder/logrank-%J.stdout \
        #          -err $checkpoints_folder/logrank-%J.stderr \
        #          /bin/bash run_tp/bb_rank_all_checkpoints.sh $checkpoints_folder \
        #          | grep 'is submitted to queue')

        # echo $jbsub_info

        bash run_tp/bb_rank_all_checkpoints.sh $checkpoints_folder

    else

        echo "Done"

    fi

    echo "---------- stage 3: decoding the best and averaged models ----------"

    if [[ ! -f $checkpoints_folder/model-selection_stage3-done ]]; then

        # jbsub_info=$(jbsub \
        #  -cores 1+1 \
        #  -mem 50g \
        #  -q x86_6h \
        #  -require v100 \
        #  -name $0 \
        #  -out $checkpoints_folder/logdec_topavg-%J.stdout \
        #  -err $checkpoints_folder/logdec_topavg-%J.stderr \
        #  /bin/bash run_tp/bc_test_model_avg.sh $checkpoints_folder \
        #  | grep 'is submitted to queue')

        # echo $jbsub_info

        bash run_tp/bc_test_model_avg.sh $checkpoints_folder

    else

        echo "Done"

    fi

else

    echo "[model not trained till $epoch_last epochs]"

fi

