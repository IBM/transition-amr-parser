#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh

##### model configuration
if [ -z "$1" ]; then
    config_model=config_files/config_model_debug.sh
else
    config_model=$1
fi

set -o nounset

##### run model and evaluation with different seeds on different GPU
num_seeds=3
seeds=(42 0 315)
gpus=(0 1 2)

echo "number of seeds: $num_seeds"
echo "seeds list: ${seeds[@]}"

echo

for (( i=0; i<$num_seeds; i++ ))
do
    echo "run seed -- ${seeds[i]}"

    # set seed; this should be accepted both in config script and in running script
    seed=${seeds[i]}

    # source config: check if it changes the seed by accident, and get $MODEL_FOLDER
    . $config_model

    # check if the config script correctly set up seed
    [[ $seed != ${seeds[i]} ]] && echo "seed is not correctly set up in config file: $config_model; try setting seed default instead of fixing seed" && exit 1

    # launch the training and evaluation
    echo "GPU - ${gpus[i]}: run saved in [$MODEL_FOLDER]"

    # (CUDA_VISIBLE_DEVICES=${gpus[i]} bash run_tp/jbash_run_model-eval.sh $config_model $seed) &
    CUDA_VISIBLE_DEVICES=${gpus[i]} bash run_tp/jbash_run_model-eval.sh $config_model $seed

done

echo
