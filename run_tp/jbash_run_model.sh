#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh

##### data config
if [ -z "$1" ]; then
    config_model=run_tp/config_model_action-pointer.sh
else
    config_model=$1
fi

seed=${2:-42}

# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $config_model

# this is set in the above file sourced
# set -o nounset

echo

##### run the job directly (e.g. in interactive mode)

if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

    echo -e "\nRun training ---"
    echo [$MODEL_FOLDER]

    # this is necessary for output redirected to file
    mkdir -p $MODEL_FOLDER

    # "|& tee file" will dump output to file as well as to terminal
    # "&> file" only dumps output to file
    # interactive: debug
    # /bin/bash $dir/run_model_action-pointer.sh $config_model $seed #|& tee $MODEL_FOLDER/log.train

    # formal run: send to background
    /bin/bash $dir/run_model_action-pointer.sh $config_model $seed &> $MODEL_FOLDER/log.train &
    now=$(date +"[%T - %D]")
    echo "$now train - PID - $!: $MODEL_FOLDER" >> .jbsub_logs/pid_model-folder.history

    echo "Log for training written at $MODEL_FOLDER/log.train"

    # on CCC, but not taking care of log locations inside the $MODEL_FOLDER
    # bash_x86_12h_v100 $dir/run_model_action-pointer.sh $config_model $seed |& tee $MODEL_FOLDER/log.train

fi

echo
