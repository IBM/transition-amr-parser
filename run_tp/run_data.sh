#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset


##### root folder to store everything
rootdir=EXP

##### load data config
config_data=config_data_o3align.sh
dir=$(dirname $0)
. $dir/$config_data $rootdir   # we should always call from one level up
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

echo "[Data directories:]"
echo $ORACLE_FOLDER
echo $DATA_FOLDER


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

cp $dir/$config_data $ROOTDIR/$DATADIR/config_data.sh

# exit 0

##### optional: test the oracle smatch for dev and test set
echo "[Testing oracle Smatch:]"
. $dir/amr_oracle_smatch.sh ""