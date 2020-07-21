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
MAX_WORDS=100

##### ORACLE EXTRACTION
# Given sentence and aligned AMR, provide action sequence that generates the
# AMR back
[ -d $ORACLE_FOLDER ] && echo "Directory to oracle $ORACLE_FOLDER already exists." && exit 0

mkdir -p $ORACLE_FOLDER

python transition_amr_parser/o7_data_oracle.py \
    --in-amr $AMR_TRAIN_FILE \
    --out-sentences $ORACLE_FOLDER/train.en \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-rule-stats $ORACLE_FOLDER/train.rules.json \
    --multitask-max-words $MAX_WORDS  \
    --out-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action

python transition_amr_parser/o7_data_oracle.py \
    --in-amr $AMR_DEV_FILE \
    --out-sentences $ORACLE_FOLDER/dev.en \
    --out-actions $ORACLE_FOLDER/dev.actions \
    --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
    --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action

python transition_amr_parser/o7_data_oracle.py \
    --in-amr $AMR_TEST_FILE \
    --out-sentences $ORACLE_FOLDER/test.en \
    --out-actions $ORACLE_FOLDER/test.actions \
    --out-rule-stats $ORACLE_FOLDER/test.rules.json \
    --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action
