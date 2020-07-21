#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### CONFIG
dir=$(dirname $0)
if [ -z "$1" ]; then
    config="config.sh"
else
    config=$1
fi
. $dir/$config    # we should always call from one level up


##### script specific config

##### PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq
# format
rm -Rf $DATA_FOLDER
mkdir -p $DATA_FOLDER
python fairseq_ext/preprocess.py \
    --user-dir ../fairseq_ext \
    --task amr_pointer \
    --source-lang en \
    --target-lang actions \
    --trainpref $ORACLE_FOLDER/train \
    --validpref $ORACLE_FOLDER/dev \
    --testpref $ORACLE_FOLDER/test \
    --destdir $DATA_FOLDER \
    --workers 1 \
#     --pretrained-embed $PRETRAINED_EMBED \
#     --machine-type AMR \
#     --machine-rules $ORACLE_FOLDER/train.rules.json
