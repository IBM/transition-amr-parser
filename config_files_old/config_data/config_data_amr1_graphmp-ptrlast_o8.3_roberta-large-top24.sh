#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
. set_exps.sh    # general setup for experiments management (save dir, etc.)
set -o nounset

# set root directory
# (the config script should be called within other scripts, where data root dir is defined)
if [ -z ${ROOTDIR+x} ]; then
    ROOTDIR=EXP
fi


##### Paths to used data
# note that training data is a LDC2016 preprocessed using Tahira's scripts
LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/
# This contains 2017 data preprocessed follwoing transition_amr_parser README
# it is needed to pass the dev oracle tests
LDC2017_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2017T10_preprocessed_TAP_v0.0.1/

# For AMR1.0
LDC2014_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/AMR_1.0/

##### AMR original data

### o3 version alignments
# AMR_TRAIN_FILE=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
# the above data has errors in alignments (o3-prefix)

# AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/kaln_2016.txt.mrged
# AMR_DEV_FILE=$LDC2017_AMR_CORPUS/dev.txt
# AMR_TEST_FILE=$LDC2017_AMR_CORPUS/test.txt

### o5 version alignments
# AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/psuedo.txt
# the above data has errors in alignments (o5-prefix)

# AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/jkaln.txt
# AMR_DEV_FILE=$LDC2016_AMR_CORPUS/dev.txt.removedWiki.noempty.JAMRaligned
# AMR_TEST_FILE=$LDC2016_AMR_CORPUS/test.txt.removedWiki.noempty.JAMRaligned

AMR_TRAIN_FILE=amr_corpus/amr1.0/AMR_1.0_train_jkaln_pseudo.txt
AMR_DEV_FILE=amr_corpus/amr1.0/AMR_1.0_dev_jaln.txt
AMR_TEST_FILE=amr_corpus/amr1.0/AMR_1.0_test_jaln.txt


### WIKI files
# NOTE: If left empty no wiki will be added
WIKI_DEV=""
AMR_DEV_FILE_WIKI=""
WIKI_TEST=""
AMR_TEST_FILE_WIKI=""

# NOTE: original dev set also same as below
# AMR_DEV_FILE_WIKI=$LDC2016_AMR_CORPUS/dev.txt

##### CONFIG
TASK=amr_action_pointer_graphmp    # amr_action_pointer_graphmp_amr1 for not interrupting running decoding process
SWAP_ARC_FOR_NODE=0    # for target input, do not substitute the arc labels with node actions
# NOTE currently this flag is not set (it's manually switched)
# NOTE this would also affect decoding! Modify decoding as well to use this
MAX_WORDS=0

ORACLEDIR=data_amr1/graphmp-ptrlast_o8.3_act-states
EMBDIR=data_amr1/en_embeddings

ORACLE_FOLDER=$ROOTDIR/$ORACLEDIR/oracle            # oracle actions, etc.
DATA_FOLDER=$ROOTDIR/$ORACLEDIR/processed           # preprocessed actions states information, etc.
EMB_FOLDER=$ROOTDIR/$EMBDIR/roberta_large_top24     # pre-stored pretrained en embeddings (not changing with oracle)

ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"

# PRETRAINED_EMBED=roberta.base
# PRETRAINED_EMBED_DIM=768
# BERT_LAYERS=12

PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
