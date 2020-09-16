#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh

# ##### data config
if [ -z "$1" ]; then
    config_model=run_tp/config_model_action-pointer.sh
else
    config_model=$1
fi

seed=${2:-42}

# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

# could be used as the job name
config_tag="$(basename $config_model | sed 's@config_model_\(.*\)\.sh@\1@g')"

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $config_model

# this is set in the above file sourced
# set -o nounset

echo

##### submit the training job to ccc
if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

echo -e "\nRun training ---"
echo [$MODEL_FOLDER]

##### submit the training job to ccc
jbsub_tag=log
train_queue=x86_12h
gpu_type=v100

mkdir -p $MODEL_FOLDER

jbsub_info=$(jbsub \
             -cores 1+1 \
             -mem 50g \
             -q $train_queue \
             -require $gpu_type \
             -name run_model_action-pointer \
             -out $MODEL_FOLDER/${jbsub_tag}-%J.stdout \
             -err $MODEL_FOLDER/${jbsub_tag}-%J.stderr \
             /bin/bash $dir/run_model_action-pointer.sh $config_model $seed \
             | grep 'is submitted to queue')


# Get job ID
echo $jbsub_info
jid=$(
    echo $jbsub_info \
    | sed 's@Job <\([0-9]*\)> is submitted to queue .*@\1@' \
)

echo "Log for training written at $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout / .stderr"

# record job id and corresponding model checkpoints folder for debug checks
echo "train - jobID - <$jid>: $MODEL_FOLDER" >> .jbsub_logs/jid_model-folder.history

# Tail logs to command line
# echo "Waiting for $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout"
# while true; do
#     [ -f "$MODEL_FOLDER/${jbsub_tag}-${jid}.stdout" ] && break
#     sleep 1
# done

# tail -f $MODEL_FOLDER/${jbsub_tag}-${jid}.std*

fi


##### submit the evaluation job to ccc

# if [[ -f $MODEL_FOLDER/model-selection_stage3-done ]]; then

#     echo "Model selection and final testing all done --- do nothing."

# else

# echo -e "\nRun evaluation ---"
# echo [$MODEL_FOLDER]

# # disallow the print and send the command to background
# # /bin/bash $dir/jbsub_run_eval.sh $config_model $seed &> /dev/null &
# /bin/bash $dir/jbsub_run_eval_updated.sh $config_model $seed &> $MODEL_FOLDER/logeval.launch &
# echo "Log for launching the evaluation and model selection written at $MODEL_FOLDER/logeval.launch"
# # record pid for debug and kill checks
# echo "launch eval - PID - $!: $MODEL_FOLDER" >> .jbsub_logs/jid_model-folder.history

# fi

# echo

