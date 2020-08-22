#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh

# ##### data config
if [ -z "$1" ]; then
    config_model=config_model_action-pointer.sh
else
    config_model=$1
fi

seed=${2:-42}

# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $dir/$config_model

# this is set in the above file sourced
# set -o nounset


##### submit the job to ccc
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

# Tail logs to command line
echo "Waiting for $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout"
while true; do
    [ -f "$MODEL_FOLDER/${jbsub_tag}-${jid}.stdout" ] && break
    sleep 1
done

tail -f $MODEL_FOLDER/${jbsub_tag}-${jid}.std*

