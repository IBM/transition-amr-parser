#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset


##### root folder to store everything
rootdir=EXP

##############################################################

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

# exit 0
###############################################################

##### model configuration
shift_pointer_value=1
tgt_vocab_masks=0
share_decoder_embed=1
seed=42
max_epoch=120

expdir=exp_o3align_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}_tie${share_decoder_embed}    # action-pointer
# expdir=exp_o3align_act-pos_vmask${tgt_vocab_masks}    # action-pointer
# expdir=exp_o3align_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}    # action-pointer
MODEL_FOLDER=EXP/$expdir/models_seed$seed

echo "[Training:]"
. $dir/ac_train.sh

cp $dir/$config_data EXP/$expdir/config_data.sh
cp $0 $MODEL_FOLDER/
cp $dir/ac_train.sh $MODEL_FOLDER/train.sh

# exit 0
###############################################################

##### decoding configuration
model_epoch=_last
# beam_size=1
batch_size=128

echo "[Decoding and computing smatch:]"
for beam_size in 1 5 10
do
    . $dir/ad_test.sh "" dev
    . $dir/ad_test.sh "" test
done

cp $dir/ad_test.sh $MODEL_FOLDER/test.sh
