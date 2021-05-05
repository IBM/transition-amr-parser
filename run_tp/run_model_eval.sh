#!/bin/bash

# evaluation of checkpoints, model selection
# and clean checkpoints

set -o errexit
set -o pipefail
# . set_environment.sh

##### root folder to store everything
ROOTDIR=/dccstor/jzhou1/work/EXP


##############################################################

##### load model config
if [ -z "$1" ]; then
    config_model=run_tp/config_model_action-pointer.sh
else
    config_model=$1
fi

seed=$2

set -o nounset

dir=$(dirname $0)
. $config_model   # $config_model should always include its path
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $EMB_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM
#
# $MODEL_FOLDER    # need $ROOTDIR set
# $eval_init_epoch

eval_init_epoch=${eval_init_epoch:-81}
# for debug
# eval_init_epoch=8

###############################################################

##### model selection decoding configuration
beam_size=1
batch_size=128
use_pred_rules=0

# if [ "$WIKI_DEV" == "" ]; then
# consider both AMR 1.0 and AMR 3.0 with blink
if [[ "$WIKI_DEV" == "" ]] && [[ ${BLINK_CACHE_PATH:-""} == "" ]]; then
    score_name=smatch
else
    score_name=wiki.smatch
fi
# best and average checkpoints'name are also associated with $score_name (e.g. checkpoint_wiki-smatch_*.pt)
score_name_tag=$(echo $score_name | sed 's/\./-/g')


##### functions for evaluation process
function eval_one_checkpoint {
    # decoding for one checkpoint, and get the smatch scores
    # then do model selection and link best models
    local test_model=$1

    echo -e "\n$test_model"
    echo "[Decoding and computing smatch:]"

    # or use the $MODEL_FOLDER from outside of the function set in the script
    MODEL_FOLDER=$(dirname $test_model)

    # decoding setup
    model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
    # beam_size=1
    # batch_size=128
    # use_pred_rules=0

    # run decoding
    . run_tp/ad_test.sh "" dev

    # this will reulst in:
    # "$MODEL_FOLDER/beam${beam_size}/valid_checkpoint${model_epoch}.wiki.smatch" (when wiki is provided)
    # or "$MODEL_FOLDER/beam${beam_size}/valid_checkpoint${model_epoch}.smatch" (when wiki is not provided)

}


function eval-new-checkpoint_and_rank-link-remove {
    # one step of:
    # a) evaluating new checkpoints (one or many) that has not been evaluated
    # b) model selection: based on all previous decoding results,
    #    rank the models, link the 5 best models, and remove unuseful checkpoints
    local checkpoints_folder=$1

    for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r); do

        epoch=$(basename $test_model | sed 's@checkpoint\(.*\)\.pt@\1@g')

        if (( $epoch >= $eval_init_epoch )); then

            # decoding if not done before
            if [[ ! -f "$checkpoints_folder/beam${beam_size}/valid_checkpoint${epoch}.${score_name}" ]]; then

                eval_one_checkpoint $test_model

            fi

        fi

    done

    # sanity check (this error should not raise)
    [[ ! -d $checkpoints_folder/beam${beam_size} ]] \
    && echo "Decoding results folder [$checkpoints_folder/beam${beam_size}] does not exist"  \
    && return 0    # NOTE with "set -e" return any value not 0 will cause the whole script to exit

    # rank the models, link the best 5 models, and remove unuseful checkpoints
    echo "rank, link, and remove --- "
    python run_tp/be_rank-link-remove.py \
        --checkpoints $checkpoints_folder \
        --link_best 5 \
        --score_name $score_name \
        --remove 1

}


function average_best_checkpoints {
    # average best checkpoints
    local checkpoints_folder=$1

    if [[ -f $checkpoints_folder/checkpoint_${score_name_tag}_best3.pt ]]; then

        python fairseq_ext/average_checkpoints.py \
                --input \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best1.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best2.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best3.pt \
                --output $checkpoints_folder/checkpoint_${score_name_tag}_top3-avg.pt

    fi


    if [[ -f $checkpoints_folder/checkpoint_${score_name_tag}_best5.pt ]]; then

        python fairseq_ext/average_checkpoints.py \
                --input \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best1.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best2.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best3.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best4.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best5.pt \
                --output $checkpoints_folder/checkpoint_${score_name_tag}_top5-avg.pt

    fi

}


function eval_best_avg_models {
    # decoding and test best averaged models
    local checkpoints_folder=$1

    # if [[ -f $checkpoints_folder/checkpoint_${score_name_tag}_best1.pt ]]; then

    #     test_model=$checkpoints_folder/checkpoint_${score_name_tag}_best1.pt

    #     echo -e "\n$test_model"
    #     echo "[Decoding and computing smatch:]"

    #     # or use the $MODEL_FOLDER from outside of the function set in the script
    #     # MODEL_FOLDER=$checkpoints_folder
    #     # or: MODEL_FOLDER=$(dirname $test_model)

    #     # decoding setup
    #     model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
    #     # beam_size=1
    #     # batch_size=128
    #     # use_pred_rules=0

    #     # run decoding
    #     for beam_size in 1 5 10
    #     do
    #         if [ ! -s $checkpoints_folder/beam${beam_size}/valid_checkpoint${model_epoch}.${score_name} ]; then
    #         . run_tp/ad_test.sh "" dev
    #         fi
    #         if [ ! -s $checkpoints_folder/beam${beam_size}/test_checkpoint${model_epoch}.${score_name} ]; then
    #         . run_tp/ad_test.sh "" test
    #         fi
    #     done

    # fi


    ##### Loop over existing averaged checkpoints
    for test_model in $(find $checkpoints_folder -iname "checkpoint_${score_name_tag}_top[0-9]*-avg.pt" | sort -r); do

        echo -e "\n$test_model"
        echo "[Decoding and computing smatch:]"

        # or use the $MODEL_FOLDER from outside of the function set in the script
        # MODEL_FOLDER=$checkpoints_folder
        # or: MODEL_FOLDER=$(dirname $test_model)

        # decoding setup
        model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
        # beam_size=1
        # batch_size=128
        # use_pred_rules=0

        # run decoding
        for beam_size in 1 5 10
        do
            if [ ! -s $checkpoints_folder/beam${beam_size}/valid_checkpoint${model_epoch}.${score_name} ]; then
            . run_tp/ad_test.sh "" dev
            fi
            if [ ! -s $checkpoints_folder/beam${beam_size}/test_checkpoint${model_epoch}.${score_name} ]; then
            . run_tp/ad_test.sh "" test
            fi
        done

    done

    ##### then do the single best model
    if [[ -f $checkpoints_folder/checkpoint_${score_name_tag}_best1.pt ]]; then

        test_model=$checkpoints_folder/checkpoint_${score_name_tag}_best1.pt

        echo -e "\n$test_model"
        echo "[Decoding and computing smatch:]"

        # or use the $MODEL_FOLDER from outside of the function set in the script
        # MODEL_FOLDER=$checkpoints_folder
        # or: MODEL_FOLDER=$(dirname $test_model)

        # decoding setup
        model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
        # beam_size=1
        # batch_size=128
        # use_pred_rules=0

        # run decoding
        for beam_size in 1 5 10
        do
            if [ ! -s $checkpoints_folder/beam${beam_size}/valid_checkpoint${model_epoch}.${score_name} ]; then
            . run_tp/ad_test.sh "" dev
            fi
            if [ ! -s $checkpoints_folder/beam${beam_size}/test_checkpoint${model_epoch}.${score_name} ]; then
            . run_tp/ad_test.sh "" test
            fi
        done

    fi

}


###############################################################

# NOTE $max_epoch is set for training; do not use that name
max_saved_epoch=0

function get_max_saved_epoch {
    # get the max epoch number among checkpoints so far
    local checkpoints_folder=$1

    for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r); do

        epoch=$(basename $test_model | sed 's@checkpoint\(.*\)\.pt@\1@g')

        if (( $epoch > $max_saved_epoch )); then
            max_saved_epoch=$epoch
        fi

        # NOTE do not use this at the end of this for loop; it would cause script to exit (why?)
        # (( $epoch > $max_saved_epoch )) && max_saved_epoch=$epoch

    done
}

##### listen to the checkpoints folder: do evaluation whenver a new checkpoint is saved

##### to get the time elapsed --> time out the script automatically ###########################
# time out condition:
# when the $max_saved_epoch does not increase in the checkpoints folder after certain time.
# NOTE
# a) the $time_max here should be set longer than the time for one epoch training,
#    across machine K80, V100, etc.
# b) if this evaluation script is called during training, this would exit if training fails,
#    i.e. no new checkpoint saved after a long time (but not after the $max_epoch)
# c) if this evaluation script is called after the training is done, this would exit if the
#    training was incomplete, i.e. the $max_saved_epoch is less than supposed $max_epoch, and
#    nothing would happen after a while
# NOTE
# d ) we never remove the last checkpoint with the epoch number, otherwise the logic would fail
################################################################################################
echo -e "\nRunning [$0]"

cp $0 $MODEL_FOLDER/

# time_max=5
time_max_between_epochs=${time_max_between_epochs:-30}
# time_max=$(( 60 * 15 ))    # 15 mins
# time_max=$(( 60 * 30 ))    # 30 mins
time_max=$(( 60 * $time_max_between_epochs ))    # 30 mins
echo "----- [max waiting time between epochs: $time_max seconds ($(( $time_max / 60 )) mins)] -----"
start=$SECONDS


echo -e "\n[Evaluation and model selection and final testing]"
echo -e "$MODEL_FOLDER"

echo "---------- stage 1: decoding all checkpoints and ranking ----------"

if [[ ! -f $MODEL_FOLDER/model-selection_stage1-done ]]; then

    while true; do
        prev_max_saved_epoch=$max_saved_epoch
        get_max_saved_epoch $MODEL_FOLDER

        # debug
        # echo $prev_max_saved_epoch
        # echo $max_saved_epoch
        # echo $eval_init_epoch
        # echo $max_epoch

        # if there is a new checkpoint && the new checkpoint is in the range of evaluation epochs
        # do decoding and model selection
        (( $max_saved_epoch > $prev_max_saved_epoch && $max_saved_epoch >= eval_init_epoch )) \
        && eval-new-checkpoint_and_rank-link-remove $MODEL_FOLDER

        if (( $max_saved_epoch == $max_epoch )); then

            echo "[Finished] stage 1: decoding all checkpoints"
            touch $MODEL_FOLDER/model-selection_stage1-done

            break

        fi


        ##### time out exit: to handle the training failure or incomplete cases
        if (( $max_saved_epoch > $prev_max_saved_epoch )); then

            start=$SECONDS

        else

            end=$SECONDS
            duration=$(( $end - $start ))
            (( $duration > $time_max )) && echo "time out between epochs ($duration seconds)" && exit 1

        fi

        # here the sleep time should be reasonable during training: not too long for checkpoints to accumulate
        sleep 1

    done

else

    echo "Done"

fi


echo "---------- stage 2: get averaged models ----------"

if [[ ! -f $MODEL_FOLDER/model-selection_stage2-done ]]; then

    average_best_checkpoints $MODEL_FOLDER
    echo "[Finished] stage 2: rank all checkpoints and get averaged models"
    touch $MODEL_FOLDER/model-selection_stage2-done

else

    echo "Done"

fi


echo "---------- stage 3: decoding the best and averaged models ----------"

if [[ ! -f $MODEL_FOLDER/model-selection_stage3-done ]]; then

    eval_best_avg_models $MODEL_FOLDER
    echo "[Finished] stage 3: decoding the best and averaged models"
    touch $MODEL_FOLDER/model-selection_stage3-done

else

    echo "Done"

fi
