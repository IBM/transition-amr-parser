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

echo "[Configuration file:]"
echo $config
. $config # $config should include its path
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

echo "[Data directories:]"
echo $ORACLE_FOLDER
echo $DATA_FOLDER
echo $EMB_FOLDER

exit

##### preprocess data (will do nothing if data exists)
echo "[Building oracle actions:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. scripts/action-pointer/aa_amr_actions.sh ""

echo "[Preprocessing data:]"
. scripts/action-pointer/ab_preprocess.sh ""

# change path to original data as we have copied in processing
AMR_TRAIN_FILE=$ORACLE_FOLDER/ref_train.amr
AMR_DEV_FILE=$ORACLE_FOLDER/ref_dev.amr
AMR_TEST_FILE=$ORACLE_FOLDER/ref_test.amr

cp $config $ROOTDIR/$ORACLEDIR/config_data.sh

# exit 0

##### optional: test the oracle smatch for dev and test set
echo "[Testing oracle Smatch:]"
. scripts/action-pointer/amr_oracle_smatch.sh ""
