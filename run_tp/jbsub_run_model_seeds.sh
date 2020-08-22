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

config_tag="$(basename $config_model | sed 's@config_model_\(.*\)\.sh@\1@g')"


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

num_seeds=3
seeds=(43 44 0)

echo "number of seeds: $num_seeds"
echo "seeds list: ${seeds[@]}"

echo

for (( i=0; i<$num_seeds; i++ ))
do
    echo "run seed -- ${seeds[i]}"
    
    # set seed; this should be accepted both in config script and in running script
    seed=${seeds[i]}
    
    # source config: for $MODEL_FOLDER to exist to save logs
    . $dir/$config_model
    
    # check if the config script correctly set up seed
    [[ $seed != ${seeds[i]} ]] && echo "seed is not correctly set up in config file: $config_model; try setting seed default instead of fixing seed" && exit 1
    
    # make model directory to save logs
    mkdir -p $MODEL_FOLDER
    
    # submit job on CCC
    jbsub_info=$(jbsub \
                 -cores 1+1 \
                 -mem 50g \
                 -q $train_queue \
                 -require $gpu_type \
                 -name ${config_tag}_seed${seed} \
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
    echo "Log written at $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout"

done

echo

# Tail logs to command line
echo "Waiting for $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout"
while true; do
    [ -f "$MODEL_FOLDER/${jbsub_tag}-${jid}.stdout" ] && break
    sleep 1
done

tail -f $MODEL_FOLDER/${jbsub_tag}-${jid}.std*

