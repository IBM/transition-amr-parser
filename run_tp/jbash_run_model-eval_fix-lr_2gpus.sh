#!/bin/bash

set -o errexit
set -o pipefail
. set_environment.sh

##### data config
if [ -z "$1" ]; then
    config_model=run_tp/config_model_action-pointer.sh
else
    config_model=$1
fi

seed=${2:-42}

gpus=($3 $4)

# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $config_model

# this is set in the above file sourced
# set -o nounset

echo

##### run the training job directly (e.g. in interactive mode)
if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

    echo -e "\nRun training ---"
    echo [$MODEL_FOLDER]

    # this is necessary for output redirected to file
    mkdir -p $MODEL_FOLDER

    # formal run: send to background
    CUDA_VISIBLE_DEVICES=${gpus[0]} /bin/bash $dir/run_model_action-pointer_fix-lr.sh $config_model $seed &> $MODEL_FOLDER/log.train &

    # record job process id and corresponding model checkpoints folder for debug checks
    now=$(date +"[%T - %D]")
    echo "$now train - PID - $!: $MODEL_FOLDER" >> .jbsub_logs/pid_model-folder.history

    echo "Log for training written at $MODEL_FOLDER/log.train"

    # Tail logs to command line
    # echo "Waiting for $MODEL_FOLDER/log.train"
    # while true; do
    #     [ -f "$MODEL_FOLDER/log.train" ] && break
    #     sleep 1
    # done

    # tail -f $MODEL_FOLDER/log.train

fi


##### run the evaluation job directly (e.g. in interactive mode)
if [[ -f $MODEL_FOLDER/model-selection_stage3-done ]]; then

    echo "Model selection and final testing all done --- do nothing."

else

    echo -e "\nRun evaluation ---"
    echo [$MODEL_FOLDER]

    # disallow the print and send the command to background
    # /bin/bash $dir/jbash_run_eval_inner.sh $config_model $seed &> /dev/null &
    CUDA_VISIBLE_DEVICES=${gpus[1]} /bin/bash $dir/jbash_run_eval_inner.sh $config_model $seed &> $MODEL_FOLDER/logeval.launch &
    echo "Log for launching the evaluation and model selection written at $MODEL_FOLDER/logeval.launch"
    # record pid for debug and kill checks
    now=$(date +"[%T - %D]")
    echo "$now launch eval - PID - $!: $MODEL_FOLDER" >> .jbsub_logs/pid_model-folder.history

fi

echo
