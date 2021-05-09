#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh


##### root folder to store everything
ROOTDIR=/dccstor/jzhou1/work/EXP

##############################################################

##### load model config
if [ -z "$1" ]; then
    config_model=run_tp/config_model_action-pointer.sh
else
    config_model=$1
fi

seed=$2

set -o nounset

dir=$(dirname $0)
. $config_model   # $config_model should always include its path
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $EMB_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM

##############################################################

#### data

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
# AMR_TRAIN_FILE=$ORACLE_FOLDER/ref_train.amr
# AMR_DEV_FILE=$ORACLE_FOLDER/ref_dev.amr
# AMR_TEST_FILE=$ORACLE_FOLDER/ref_test.amr

# exit 0
###############################################################

##### train model (will do nothing if $MODEL_FOLDER exists)

echo "[Training:]"

mkdir -p $MODEL_FOLDER

cp $config_data $ROOTDIR/$expdir/ || true
# to skip cp error (e.g. when $config_model already exists and cp the same file)
cp $config_model $MODEL_FOLDER/ || true

# change the seed name in the particular model configuration copied
sed -i "s/seed:-42/seed:-${seed}/g" $MODEL_FOLDER/$(basename $config_model)

cp $0 $MODEL_FOLDER/
cp $dir/ac_train.sh $MODEL_FOLDER/train.sh

. $dir/ac_train.sh

# exit 0
###############################################################

##### decoding configuration
model_epoch=_last
# beam_size=1
batch_size=128
use_pred_rules=0

# echo "[Decoding and computing smatch:]"
# for beam_size in 1 5 10
# do
#     . $dir/ad_test.sh "" dev
#     . $dir/ad_test.sh "" test
# done

cp $dir/ad_test.sh $MODEL_FOLDER/test.sh
