#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
config=$1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# Require aligned AMR
[ ! -f "$ORACLE_FOLDER/.done" ] && \
    echo -e "\nRequires oracle in $ORACLE_FOLDER\n" && \
    exit 1

##### PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq format

TASK=${TASK:-amr_action_pointer}

if [[ (-f $DATA_FOLDER/.done) && (-f $EMB_FOLDER/.done) ]]; then

    echo "Directory to processed oracle data: $DATA_FOLDER"
    echo "and source pre-trained embeddings: $EMB_FOLDER"
    echo "already exists --- do nothing."

else

    if [[ $TASK == "amr_action_pointer" ]]; then

        python fairseq_ext/preprocess.py \
            --user-dir ../fairseq_ext \
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
            --user-dir ../fairseq_ext \
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
            --user-dir ../fairseq_ext \
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


    else

        echo -e "Unknown task $TASK"
        exit 1

    fi

fi
