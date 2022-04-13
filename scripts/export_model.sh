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

model_folder=${MODEL_FOLDER}-seed${seed}/
model_name=$(basename $config .sh)

# needed files
checkpoint=$model_folder/$DECODING_CHECKPOINT

echo "$checkpoint"

[ ! -f "$checkpoint" ] && echo "Is $config training complete?" && exit 1

# remove optimizer from checkpoint
python scripts/remove_optimizer_state.py $checkpoint $model_folder/model.pt

# zip all
zip -r ${model_name}.zip \
    $model_folder/model.pt \
    $model_folder/config.sh \
    $model_folder/dict.actions_nopos.txt \
    $model_folder/dict.en.txt \
    $model_folder/machine_config.json
