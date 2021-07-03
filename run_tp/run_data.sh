#!/bin/bash

set -o errexit
set -o pipefail
. set_environment.sh
# set -o nounset


##### root folder to store everything
ROOTDIR=/dccstor/jzhou1/work/EXP

##### load data config
dir=$(dirname $0)
if [ -z "$1" ]; then
    config_data=$dir/config_data/config_data_o3_roberta-base-last.sh
else
    config_data=$1
fi

echo "[Data configuration file:]"
echo $config_data


if [ ! -z "$2" ]; then
    ROOTDIR=$2
fi

# NOTE: this should be set after the "$1" check, otherwise it will throw error "unbound variable" if $1 is not set!!
set -o nounset


. $config_data    # $config_data should include its path
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

echo "[Data directories:]"
echo $ORACLE_FOLDER
echo $DATA_FOLDER
echo $EMB_FOLDER


##### preprocess data (will do nothing if data exists)
echo "[Building oracle actions:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. $dir/aa_amr_actions.sh ""

echo "[Preprocessing data:]"
. $dir/ab_preprocess.sh ""

# change path to original data as we have copied in processing
AMR_TRAIN_FILE=$ORACLE_FOLDER/ref_train.amr
AMR_DEV_FILE=$ORACLE_FOLDER/ref_dev.amr
AMR_TEST_FILE=$ORACLE_FOLDER/ref_test.amr

cp $config_data $ROOTDIR/$ORACLEDIR/config_data.sh

# exit 0

##### optional: test the oracle smatch for dev and test set
echo "[Testing oracle Smatch:]"
. $dir/amr_oracle_smatch.sh ""
