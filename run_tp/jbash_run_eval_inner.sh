#!/bin/bash

# evaluation of checkpoints, model selection
# and clean checkpoints

set -o errexit
set -o pipefail
# . set_environment.sh

##### root folder to store everything
. set_exps.sh    # set $ROOTDIR

if [ -z ${ROOTDIR+x} ]; then
    ROOTDIR=/dccstor/jzhou1/work/EXP
fi

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

# # NOTE set the evaluation starting epoch to be a bit later
# #      than the first eval epoch, to reduce waiting during
# #      training and keep evaluation job within 6 hours on CCC
# eval_init_epoch=101


###############################################################
# waiting to launch the script

# echo -e "\nwaiting to launch evaluation process (due to resource occupied)"

# sleep 2h

# echo -e "\n"

###############################################################


###############################################################
# to get the time elapsed --> time out the script automatically
# time_max=5
time_max=$(( 3600 * 12 ))    # 12 hours, in interactive mode
echo "----- [max waiting time $time_max seconds ($(( $time_max / 3600 )) hours)] -----"
start=$SECONDS

# wait for the model checkpoint folder to appear
# this is necessary for the "find" command below (otherwise an error will raise)
echo -e "\nwaiting for [$MODEL_FOLDER] to appear"
while true; do
    [[ -d $MODEL_FOLDER ]] && echo "--- Got it!" && break
    sleep 2

    # time out
    end=$SECONDS
    duration=$(( end - start ))
    (( $duration > $time_max )) && echo "time out ($duration seconds)" && exit 1
done

# listen for the first epoch to evaluationn to appear, which
# triggers the whole evaluation protocol
echo -e "\nwaiting for the first checkpoint to eval: checkpoint${eval_init_epoch}"
while true; do

    # for first run: if the first checkpoint to eval start to appear
    if [[ -f $MODEL_FOLDER/checkpoint${eval_init_epoch}.pt ]]; then

        echo "--- Triggered - first checkpoint to eval appears"
        break

    else

    # for non-first run: the first checkpoint to eval may have been deleted
    # check if later checkpoints exit
        for test_model in $(find $MODEL_FOLDER -iname 'checkpoint[0-9]*.pt' | sort -r); do
            epoch=$(basename $test_model | sed 's@checkpoint\(.*\)\.pt@\1@g')
            if (( $epoch > $eval_init_epoch )); then
                # must use (( )) instead of [[ ]] for numerical comparison,
                # instead of string comparison
                echo "--- Triggered - first checkpoint not existing, but later checkpoint${epoch} exists"
                # break 2 levels of loops
                break 2
            fi
        done

    fi

    sleep 5

    # time out
    end=$SECONDS
    duration=$(( end - start ))
    (( $duration > $time_max )) && echo "time out ($duration seconds)" && exit 1

done


###############################################################
# launch the evaluation process

# "|& tee file" will dump output to file as well as to terminal
# "&> file" only dumps output to file
# interactive: debug
# /bin/bash $dir/run_model_eval.sh $config_model $seed #|& tee $MODEL_FOLDER/log.eval

# formal run: send to background

# maintain the previous evaluation log files in case of rerunning due to some issues
if [[ -f $MODEL_FOLDER/log.eval ]]; then
    n=0
    while [[ -f $MODEL_FOLDER/log.eval$n ]]; do
        n=$(( $n + 1 ))
    done
    mv $MODEL_FOLDER/log.eval $MODEL_FOLDER/log.eval$n
fi

/bin/bash $dir/run_model_eval.sh $config_model $seed &> $MODEL_FOLDER/log.eval &
now=$(date +"[%T - %D]")
echo "$now eval - PID - $!: $MODEL_FOLDER" >> .jbsub_logs/pid_model-folder.history

# Get pid by "$!"
