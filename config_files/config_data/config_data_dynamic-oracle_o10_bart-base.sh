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

AMR_TRAIN_FILE=amr_corpus/amr2.0/o5/jkaln.txt
AMR_DEV_FILE=amr_corpus/amr2.0/o5/dev.txt.removedWiki.noempty.JAMRaligned
AMR_TEST_FILE=amr_corpus/amr2.0/o5/test.txt.removedWiki.noempty.JAMRaligned


### WIKI files
# NOTE: If left empty no wiki will be added
WIKI_DEV=amr_corpus/amr2.0/wiki/dev.wiki
AMR_DEV_FILE_WIKI=amr_corpus/amr2.0/wiki/dev.txt
WIKI_TEST=amr_corpus/amr2.0/wiki/test.wiki
AMR_TEST_FILE_WIKI=amr_corpus/amr2.0/wiki/test.txt


# NOTE: original dev set also same as below
# AMR_DEV_FILE_WIKI=$LDC2016_AMR_CORPUS/dev.txt

##### CONFIG
TASK=amr_action_pointer_bart_dyo

ORACLEDIR=data_o10/o10_act-states
EMBDIR=data_o10/en_embeddings

ORACLE_FOLDER=$ROOTDIR/$ORACLEDIR/oracle            # oracle actions, etc.
DATA_FOLDER=$ROOTDIR/$ORACLEDIR/processed           # preprocessed actions states information, etc.
EMB_FOLDER=$ROOTDIR/$EMBDIR/bart_base     # pre-stored pretrained en embeddings (not changing with oracle)

# ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"

# PRETRAINED_EMBED=roberta.base
# PRETRAINED_EMBED_DIM=768
# BERT_LAYERS=12

PRETRAINED_EMBED=bart.base
PRETRAINED_EMBED_DIM=768
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12"
