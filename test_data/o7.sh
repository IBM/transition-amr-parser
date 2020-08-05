#!/bin/bash

set -e
set -o pipefail

# load local variables used below
# . set_environment.sh
[ -z "$1" ] && echo "$0 train (or dev)" && exit 1
test_set=$1
set -o nounset

# Configuration 
MAX_WORDS=100
# ORACLE_TAG=o7+Word${MAX_WORDS}_noclose
ORACLE_TAG=o7+Word${MAX_WORDS}
ORACLE_FOLDER=oracles/${ORACLE_TAG}

LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira
LDC2017_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2017T10_preprocessed_TAP_v0.0.1

# Select between train/dev
# train_amr=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
train_amr=/dccstor/multi-parse/transformer-amr/kaln_2016.txt.mrged
if [ "$test_set" == "dev" ];then
    # ATTENTION: To pass the tests the dev test must have alignments as those
    # obtained with the preprocessing described in README
    reference_amr=$LDC2017_AMR_CORPUS/dev.txt
    # Do not limit actions by rules
    ref_smatch=0.938
    # limit actions by rules
    ref_smatch2=0.921
elif [ "$test_set" == "train" ];then
#     reference_amr=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
    reference_amr=$train_amr
    ref_smatch=0.937
elif [ "$test_set" == "test" ]; then
    reference_amr=$LDC2017_AMR_CORPUS/test.txt
    ref_smatch=0.941
else
    echo "Usupported set $test_set"
    exit 1
fi

# TRAIN
[ ! -d $ORACLE_FOLDER/ ] && mkdir -p $ORACLE_FOLDER/

# create oracle actions from AMR and the sentence for the train set. This also
# accumulates necessary statistics in train.rules.json
if [ ! -f "$ORACLE_FOLDER/train.rules.json" ];then
    python ../transition_amr_parser/o7_data_oracle.py \
        --in-amr $train_amr \
        --out-sentences $ORACLE_FOLDER/train.en \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-rule-stats $ORACLE_FOLDER/train.rules.json \
        --multitask-max-words $MAX_WORDS  \
        --out-multitask-words $ORACLE_FOLDER/train.multitask_words \
        --copy-lemma-action  
fi

# create oracle actions from AMR and the sentence for the dev set if needed
# this is to prevent run on training data again

if [ ! -f "$ORACLE_FOLDER/${test_set}.rules.json" ];then
    python ../transition_amr_parser/o7_data_oracle.py \
        --in-amr $reference_amr \
        --out-sentences $ORACLE_FOLDER/${test_set}.en \
        --out-actions $ORACLE_FOLDER/${test_set}.actions \
        --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
        --copy-lemma-action
fi


# reconstruct AMR given sentence and oracle actions without being constrained
# by training stats
python ../transition_amr_parser/o7_fake_parse.py \
    --in-sentences $ORACLE_FOLDER/${test_set}.en \
    --in-actions $ORACLE_FOLDER/${test_set}.actions \
    --out-amr $ORACLE_FOLDER/oracle_${test_set}.amr

# evaluate reconstruction performance
# smatch="$(smatch.py --significant 3 -r 10 -f $reference_amr $ORACLE_FOLDER/oracle_${test_set}.amr)"

smatch.py --significant 3 -r 10 -f $reference_amr $ORACLE_FOLDER/oracle_${test_set}.amr

# echo "$smatch"
# echo "$ref_smatch"

