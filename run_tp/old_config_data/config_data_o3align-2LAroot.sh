#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

# set root directory
if [ -z "$1" ]; then
    ROOTDIR=EXP
else
    ROOTDIR=$1
fi

##### Paths to used data
# note that training data is a LDC2016 preprocessed using Tahira's scripts
LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/
# This contains 2017 data preprocessed follwoing transition_amr_parser README
# it is needed to pass the dev oracle tests
LDC2017_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2017T10_preprocessed_TAP_v0.0.1/

##### AMR original data

AMR_TRAIN_FILE=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
AMR_DEV_FILE=$LDC2017_AMR_CORPUS/dev.txt
AMR_TEST_FILE=$LDC2017_AMR_CORPUS/test.txt

# okay, these data are different

# AMR_TRAIN_FILE=/dccstor/multi-parse/transformer-amr/psuedo.txt
# AMR_DEV_FILE=$LDC2016_AMR_CORPUS/dev.txt.removedWiki.noempty.JAMRaligned
# AMR_TEST_FILE=$LDC2016_AMR_CORPUS/test.txt.removedWiki.noempty.JAMRaligned


##### CONFIG
DATADIR=data/o3align_roberta-base-last_act-noeos-states-2LAroot
ORACLE_FOLDER=$ROOTDIR/$DATADIR/oracle
DATA_FOLDER=$ROOTDIR/$DATADIR/processed

PRETRAINED_EMBED=roberta.base
PRETRAINED_EMBED_DIM=768

# PRETRAINED_EMBED=roberta.large
# PRETRAINED_EMBED_DIM=1024
