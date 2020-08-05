#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### CONFIG
dir=$(dirname $0)
if [ ! -z "$1" ]; then
    config=$1
    . $dir/$config    # we should always call from one level up
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

##### script specific config

##### PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq format
# [ -d $DATA_FOLDER ] && echo "Directory to processed data $DATA_FOLDER already exists." && exit 0
# rm -Rf $DATA_FOLDER

if [[ (-d $DATA_FOLDER) && (-d $EMB_FOLDER) ]]; then

    echo "Directory to processed oracle data: $DATA_FOLDER"
    echo "and source pre-trained embeddings: $EMB_FOLDER"
    echo "already exists --- do nothing."

else

    python fairseq_ext/preprocess.py \
        --user-dir ../fairseq_ext \
        --task amr_action_pointer \
        --source-lang en \
        --target-lang actions \
        --trainpref $ORACLE_FOLDER/train \
        --validpref $ORACLE_FOLDER/dev \
        --testpref $ORACLE_FOLDER/test \
        --destdir $DATA_FOLDER \
        --embdir $EMB_FOLDER \
        --workers 1 \
        --pretrained-embed $PRETRAINED_EMBED \
        --bert-layers $BERT_LAYERS
    #     --machine-type AMR \
    #     --machine-rules $ORACLE_FOLDER/train.rules.json
fi
