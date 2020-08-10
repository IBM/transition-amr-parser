#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

# set root directory
if [ -z ${ROOTDIR+x} ]; then
    ROOTDIR=/dccstor/jzhou1/work/EXP
fi


##### Paths to used data
# note that training data is a LDC2016 preprocessed using Tahira's scripts
LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/
# This contains 2017 data preprocessed follwoing transition_amr_parser README
# it is needed to pass the dev oracle tests
LDC2017_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2017T10_preprocessed_TAP_v0.0.1/

##### AMR original data

# AMR_TRAIN_FILE=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
# the above data has errors in alignments

AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/kaln_2016.txt.mrged
AMR_DEV_FILE=$LDC2017_AMR_CORPUS/dev.txt
AMR_TEST_FILE=$LDC2017_AMR_CORPUS/test.txt

# okay, these data are different

# AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/psuedo.txt
# AMR_DEV_FILE=$LDC2016_AMR_CORPUS/dev.txt.removedWiki.noempty.JAMRaligned
# AMR_TEST_FILE=$LDC2016_AMR_CORPUS/test.txt.removedWiki.noempty.JAMRaligned


##### CONFIG
MAX_WORDS=0

ORACLEDIR=data/o3_no-mw_act-states
EMBDIR=data/en_embeddings

ORACLE_FOLDER=$ROOTDIR/$ORACLEDIR/oracle            # oracle actions, etc.
DATA_FOLDER=$ROOTDIR/$ORACLEDIR/processed           # preprocessed actions states information, etc.
EMB_FOLDER=$ROOTDIR/$EMBDIR/roberta_large_top24     # pre-stored pretrained en embeddings (not changing with oracle) 


# PRETRAINED_EMBED=roberta.base
# PRETRAINED_EMBED_DIM=768
# BERT_LAYERS=12

PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
