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

# This will store the final model
mkdir -p ${MODEL_FOLDER}-seed${seed}
cp $config ${MODEL_FOLDER}-seed${seed}/config.sh

echo "[Building oracle actions:]"
mkdir -p $ORACLE_FOLDER
# TODO: replace by task agnostic oracle creation
bash run/aa_amr_actions.sh $config

echo "[Preprocessing data:]"
mkdir -p $DATA_FOLDER
bash run/ab_preprocess.sh $config

[ "$on_the_fly_decoding" = true ] \
    && echo "[Decoding and computing smatch (on the fly):]" \
    && bash run/run_model_eval.sh $config $seed &

echo "[Training:]"
bash run/ac_train.sh $config $seed 

[ "$on_the_fly_decoding" = false ] \
    && echo "[Decoding and computing smatch:]" \
    && bash run/run_model_eval.sh $config $seed
