#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\ntest_all_checkpoints.sh <checkpoints_folder>\n" && \
    exit 1
set -o nounset

# Get the data confirguration
# TODO copy the latest data configuration into the data config (done)
expdir=$(dirname $checkpoints_folder)
config_data=($expdir/config_data_*)
echo -e "\n[Experiments directory:]"
echo $expdir
echo -e "\n[Data configuration file:]"
echo $config_data
. $config_data
# now we have
# echo $AMR_TRAIN_FILE
# echo $AMR_DEV_FILE
# echo $AMR_TEST_FILE
# echo $WIKI_DEV
# echo $AMR_DEV_FILE_WIKI
# echo $WIKI_TEST
# echo $AMR_TEST_FILE_WIKI
#
# echo $DATA_FOLDER
# echo $ORACLE_FOLDER
# echo $EMB_FOLDER


##### test on the best smatch model

if [[ -f $checkpoints_folder/checkpoint_wiki-smatch_best1.pt ]]; then

    test_model=checkpoint_wiki-smatch_best1.pt

    echo -e "\n$test_model"
    echo "[Decoding and computing smatch:]"

    MODEL_FOLDER=$checkpoints_folder
    # or: MODEL_FOLDER=$(dirname $test_model)

    # decoding setup
    model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
    # beam_size=1
    batch_size=128
    use_pred_rules=0

    # run decoding
    for beam_size in 1 5 10
    do
        . run_tp/ad_test.sh "" dev
        . run_tp/ad_test.sh "" test
    done

fi


##### Loop over existing averaged checkpoints
for test_model in $(find $checkpoints_folder -iname 'checkpoint_wiki-smatch_top[0-9]*-avg.pt' | sort ); do
    
    echo -e "\n$test_model"
    echo "[Decoding and computing smatch:]"
    
    MODEL_FOLDER=$checkpoints_folder
    # or: MODEL_FOLDER=$(dirname $test_model)
    
    # decoding setup
    model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
    # beam_size=1
    batch_size=128
    use_pred_rules=0
    
    # run decoding
    for beam_size in 1 5 10
    do
        . run_tp/ad_test.sh "" dev
        . run_tp/ad_test.sh "" test
    done

done


touch $checkpoints_folder/model-selection_stage3-done
