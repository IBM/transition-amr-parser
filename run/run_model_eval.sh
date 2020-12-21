#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <checkpoints_folder>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
checkpoints_folder=$1
[ ! -d "$checkpoints_folder" ] && "Missing $checkpoints_folder" && exit 1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# extract config from checkpoint path
config=$checkpoints_folder/config.sh
[ ! -f "$config" ] && "Missing $config" && exit 1

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# determine scoring
if [ "$WIKI_DEV" == "" ]; then
    score_name=smatch
else
    score_name=wiki.smatch
fi
# best and average checkpoints'name are also associated with $score_name (e.g.
# checkpoint_wiki-smatch_*.pt)
score_name_tag=$(echo $score_name | sed 's/\./-/g')


function eval-new-checkpoint_and_rank-link-remove {
    # one step of:
    # a) evaluating new checkpoints (one or many) that has not been evaluated
    # b) model selection: based on all previous decoding results,
    #    rank the models, link the 5 best models, and remove unuseful checkpoints
    local checkpoints_folder=$1

    for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r); do

        epoch=$(basename $test_model | sed 's@checkpoint\(.*\)\.pt@\1@g')
        results_prefix=$checkpoints_folder/beam${beam_size}/valid_checkpoint${epoch}

        if (( $epoch >= $eval_init_epoch )); then

            # decoding if not done before
            if [[ ! -f "${results_prefix}.${score_name}" ]]; then
                bash run/ad_test.sh $test_model -o ${results_prefix}
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

        python fairseq/scripts/average_checkpoints.py \
                --input \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best1.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best2.pt \
                    $checkpoints_folder/checkpoint_${score_name_tag}_best3.pt \
                --output $checkpoints_folder/checkpoint_${score_name_tag}_top3-avg.pt

    fi


    if [[ -f $checkpoints_folder/checkpoint_${score_name_tag}_best5.pt ]]; then

        python fairseq/scripts/average_checkpoints.py \
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


    ##### Loop over existing averaged checkpoints
    for test_model in $(find $checkpoints_folder -iname "checkpoint_${score_name_tag}_top[0-9]*-avg.pt" | sort ); do

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
time_max=$(( 60 * 15 ))    # 15 mins
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
