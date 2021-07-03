#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

[ -z $1 ] && echo "usage: bash collect_scores_exp.sh <exp_dir>" && exit 1
set -o nounset

exp_dir=$1

# seed=0
seed=""    # check all seeds
epoch_last=120

echo -e "\n[Results for all model checkpoints under experiments:]"
echo "$exp_dir"

if [[ $seed == "" ]]; then
    model_folders=($1/models*)
else
    model_folders=($1/models_ep${epoch_last}_seed${seed})
fi


for checkpoints_folder in "${model_folders[@]}"; do

    echo $checkpoints_folder

    if [[ -f $checkpoints_folder/checkpoint${epoch_last}.pt ]]; then

        bash run_tp/collect_scores.sh $checkpoints_folder

    else

        echo "[model not trained till $epoch_last epochs]"

    fi

done
