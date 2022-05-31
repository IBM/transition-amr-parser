#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
config=$1
[ ! -f "$config" ] && "Missing $config" && exit 1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# random seed
seed=42
# decode in paralel to training. ATTENTION: you will need to GPUS for this
on_the_fly_decoding=false

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
# not using neural aligner but no alignments provided
if [ "$align_tag" != "ibm_neural_aligner" ] && [ ! -f $ALIGNED_FOLDER/.done ];then
    echo -e "\nYou need to provide $align_tag alignments\n"
    exit 1
fi

# This will store the final model
mkdir -p ${MODEL_FOLDER}seed${seed}
# Copy the config and soft-link it with an easy to find name
cp $config ${MODEL_FOLDER}seed${seed}/
rm -f ${MODEL_FOLDER}seed${seed}/config.sh
ln -s $(basename $config) ${MODEL_FOLDER}seed${seed}/config.sh

# Add a tag with the commit(s) used to train this model. 
if [ "$(git status --porcelain | grep -v '^??')" == "" ];then
    # no uncommited changes
    touch "${MODEL_FOLDER}seed${seed}/$(git log --format=format:"%h" -1)"
else
    # uncommited changes
    touch "${MODEL_FOLDER}seed${seed}/$(git log --format=format:"%h" -1)+"
fi

echo "[Aligning AMR:]"
mkdir -p $ALIGNED_FOLDER
bash run/train_aligner.sh $config

echo "[Building oracle actions:]"
mkdir -p $ORACLE_FOLDER
# TODO: replace by task agnostic oracle creation
bash run/amr_actions.sh $config

echo "[Preprocessing data:]"
mkdir -p $DATA_FOLDER
bash run/preprocess.sh $config

[ "$on_the_fly_decoding" = true ] \
    && echo "[Decoding and computing smatch (on the fly):]" \
    && bash run/run_model_eval.sh $config $seed &

echo "[Training:]"
bash run/train.sh $config $seed 

[ "$on_the_fly_decoding" = false ] \
    && echo "[Decoding and computing smatch:]" \
    && bash run/run_model_eval.sh $config $seed
