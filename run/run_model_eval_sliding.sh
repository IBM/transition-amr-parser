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

# Quick exits
# Data not extracted or aligned data not provided
if [ ! -f "$AMR_TRAIN_FILE_WIKI" ] && [ ! -f "$ALIGNED_FOLDER/train.txt" ];then
    echo -e "\nNeeds $AMR_TRAIN_FILE_WIKI or $ALIGNED_FOLDER/train.txt\n" 
    exit 1
fi
# linking cache not empty but folder does not exist
if [ "$LINKER_CACHE_PATH" != "" ] && [ ! -d "$LINKER_CACHE_PATH" ];then
    echo -e "\nNeeds linking cache $LINKER_CACHE_PATH\n"
    exit 1
fi 

# folder of the model seed
checkpoints_folder=${MODEL_FOLDER}seed${seed}/

# Evaluate all required checkpoints with EVAL_METRIC
if [ ! -f "$checkpoints_folder/epoch_tests/.done" ];then

    mkdir -p "$checkpoints_folder/epoch_tests/"

    # Note this removes models and links best models on the fly
    while [ "$(python run/status.py -c $config --seed $seed --list-checkpoints-to-eval --link-best --remove)" != "" ];do
    
        # get existing checkpoints
        ready_checkpoints=$(python run/status.py -c $config --seed $seed --list-checkpoints-ready-to-eval)
    
        # if there are no checkpoints at this moment, wait and restart loop
        if [ "$ready_checkpoints" == "" ];then
            printf "\r$$ is waiting for checkpoints of ${config}:$seed"
            sleep 1m
            continue
        fi    
        echo ""
    
        # run test for these checkpoints
        for checkpoint in $ready_checkpoints;do
            results_prefix=$checkpoints_folder/epoch_tests/dec-$(basename $checkpoint .pt)
            bash run/test_sliding.sh $checkpoint -o $results_prefix

            # clean this checkpoint. This can be helpful if we started a job
            # with lots of pending checkpoints to evaluate
            python run/status.py -c $config --seed $seed  --link-best --remove
        done
    done
    touch $checkpoints_folder/epoch_tests/.done

else

    printf "[\033[92m done \033[0m] $checkpoints_folder/epoch_tests/.done\n"
    
fi

# This should not be needed, but its a sanity check
python run/status.py -c $config --seed $seed --list-checkpoints-to-eval --link-best --remove

# 3 checkpoint average
if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt ]]; then
    echo "Evaluation/Ranking failed, missing $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt "
    exit 1
fi

if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_top3-avg.pt ]]; then
    python fairseq_ext/average_checkpoints.py \
        --input \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best1.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best2.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt \
        --output $checkpoints_folder/checkpoint_${EVAL_METRIC}_top3-avg.pt
fi


# 5 checkpoint average
if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt ]]; then
    echo "Evaluation/Ranking failed, missing $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt "
    exit 1
fi


if [[ ! -f $checkpoints_folder/checkpoint_${EVAL_METRIC}_top5-avg.pt ]]; then
    python fairseq_ext/average_checkpoints.py \
        --input \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best1.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best2.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best3.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best4.pt \
            $checkpoints_folder/checkpoint_${EVAL_METRIC}_best5.pt \
        --output $checkpoints_folder/checkpoint_${EVAL_METRIC}_top5-avg.pt
fi

# Final run
[ ! -f "$checkpoints_folder/$DECODING_CHECKPOINT" ] \
    && echo -e "Missing $checkpoints_folder/$DECODING_CHECKPOINT" \
    && exit 1
mkdir -p $checkpoints_folder/beam${BEAM_SIZE}/
bash run/test_sliding.sh $checkpoints_folder/$DECODING_CHECKPOINT -b $BEAM_SIZE
