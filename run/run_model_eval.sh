#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config> <seed>\n"
# config file
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ ! -f "$1" ] && "Missing $1" && exit 1
config=$1
# random seed
[ -z "$2" ] && echo -e "$HELP" && exit 1
seed=$2

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# folder of the model seed
checkpoints_folder=${MODEL_FOLDER}-seed${seed}/

# Evaluate all required checkpoints with EVAL_METRIC
if [ ! -f "$checkpoints_folder/epoch_tests/.done" ];then

    # TODO: Add --remove command here to remove completed checkpoints
    # TODO: Maybe also --rank here?
    while [ "$(python run/status.py $config --seed $seed --list-checkpoints-to-eval)" != "" ];do
    
        # get existing checkpoints
        ready_checkpoints=$(python run/status.py $config --seed $seed --list-checkpoints-ready-to-eval)
    
        # if there are no checkpoints at this moment, wait and restart loop
        if [ "$ready_checkpoints" == "" ];then
            printf "\rWaiting for checkpoints of ${config}:$seed"
            sleep 1m
            continue
        fi    
        echo ""
    
        # run test for these checkpoints
        for checkpoint in $ready_checkpoints;do
            results_prefix=$checkpoints_folder/epoch_tests/dec-$(basename $checkpoint .pt)
            bash run/ad_test.sh $checkpoint -o $results_prefix
        done
    done
    touch $checkpoints_folder/epoch_tests/.done
    
fi

# rank and remove
# TODO: Introduce in the code above
python run/be_rank-link-remove.py \
    --checkpoints $checkpoints_folder \
    --link_best 5 \
    --score_name $EVAL_METRIC \
    --remove 1

# 3 checkpoint average
if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt ]]; then
    echo "Evaluation/Ranking failed, missing $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt "
    exit 1
fi

python fairseq/scripts/average_checkpoints.py \
        --input \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best1.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best2.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt \
        --output $checkpoints_folder/checkpoint_${EVAL_METRIC}_top3-avg.pt


# 5 checkpoint average
if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt ]]; then
    echo "Evaluation/Ranking failed, missing $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt "
    exit 1
fi

python fairseq/scripts/average_checkpoints.py \
        --input \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best1.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best2.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best4.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt \
        --output $checkpoints_folder/checkpoint_${EVAL_METRIC}_top5-avg.pt

# Final run
[ ! -f "$checkpoints_dir/$DECODING_CHECKPOINT" ] \
    && echo -e "Missing $checkpoints_dir/$DECODING_CHECKPOINT" \
    && exit 1
bash run/ad_test.sh $checkpoints_dir/$DECODING_CHECKPOINT -b $BEAM_SIZE
