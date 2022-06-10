#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config> <seed>\n"
# config file
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ ! -f "$1" ] && "Missing $1" && exit 1
config=$1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config

[ ! -f DATA/$TASK_TAG/aligned/ibm_neural_aligner/.done ] \
    && printf "\nIs aligner training complete?\n" \
    && exit 1

zip -r ${TASK_TAG}_ibm_neural_aligner.zip \
    DATA/$TASK_TAG/aligned/ibm_neural_aligner/log/model.latest.pt \
    DATA/$TASK_TAG/aligned/ibm_neural_aligner/log/flags.json \
    DATA/$TASK_TAG/aligned/ibm_neural_aligner/vocab.text.txt \
    DATA/$TASK_TAG/aligned/ibm_neural_aligner/vocab.amr.txt \
    DATA/$TASK_TAG/aligned/ibm_neural_aligner/.done.train 

echo "Created ${TASK_TAG}_ibm_neural_aligner.zip"
