#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### CONFIG
dir=$(dirname $0)

if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config=$1
    . $config    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

##### script specific config

##### PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq format
# [ -d $DATA_FOLDER ] && echo "Directory to processed data $DATA_FOLDER already exists." && exit 0
# rm -Rf $DATA_FOLDER

TASK=${TASK:-amr_action_pointer}

if [[ (-d $DATA_FOLDER) && (-d $EMB_FOLDER) ]]; then

    echo "Directory to processed oracle data: $DATA_FOLDER"
    echo "and source pre-trained embeddings: $EMB_FOLDER"
    echo "already exists --- do nothing."

else

    if [[ $TASK == "amr_action_pointer" ]]; then

    python fairseq_ext/preprocess.py \
        --user-dir ./fairseq_ext \
        --task $TASK \
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

    elif [[ $TASK == "amr_action_pointer_graphmp" ]]; then

    python fairseq_ext/preprocess_graphmp.py \
        --user-dir ./fairseq_ext \
        --task $TASK \
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

    elif [[ $TASK == "amr_action_pointer_graphmp_amr1" ]]; then

    # a separate route of code for preprocessing of AMR 1.0 data; the only difference is in o8 state machine
    # get_valid_canonical_actions to deal with a single exmple in training set with self-loop

    python fairseq_ext/preprocess_graphmp.py \
        --user-dir ./fairseq_ext \
        --task $TASK \
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

    fi

fi
