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

config_tag="$(basename $config_model | sed 's@config_model_\(.*\)\.sh@\1@g')"


# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $config_model

# this is set in the above file sourced
# set -o nounset


##### set the random seeds and submit the job to ccc

# num_seeds=3
# seeds=(43 44 0)

num_seeds=3
seeds=(42 0 135)

# num_seeds=2
# seeds=(0 135)

echo "number of seeds: $num_seeds"
echo "seeds list: ${seeds[@]}"

echo

for (( i=0; i<$num_seeds; i++ ))
do
    echo "run seed -- ${seeds[i]}"
    
    # set seed; this should be accepted both in config script and in running script
    seed=${seeds[i]}
    
    # source config: for $MODEL_FOLDER to exist to save logs
    . $config_model
    
    # check if the config script correctly set up seed
    [[ $seed != ${seeds[i]} ]] && echo "seed is not correctly set up in config file: $config_model; try setting seed default instead of fixing seed" && exit 1
    
    # launch the training and evaluation
    bash run_tp/jbsub_run_model-eval.sh $config_model $seed

done

echo

# # Tail logs to command line
# jbsub_tag=log    # this should be the same as that in run_tp/jbsub_run_model-eval.sh

# echo "Waiting for $MODEL_FOLDER/${jbsub_tag}-${jid}.stdout"
# while true; do
#     [ -f "$MODEL_FOLDER/${jbsub_tag}-${jid}.stdout" ] && break
#     sleep 1
# done

# tail -f $MODEL_FOLDER/${jbsub_tag}-${jid}.std*

