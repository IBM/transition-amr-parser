#!/bin/bash

set -o errexit
set -o pipefail
. set_environment.sh


##### root folder to store everything
ROOTDIR=/dccstor/jzhou1/work/EXP

##############################################################

##### load model config
if [ -z "$1" ]; then
    config_model=config_model_action-pointer.sh
else
    config_model=$1
fi

seed=$2

set -o nounset

dir=$(dirname $0)
. $dir/$config_model   # we should always call from one level up
# now we have
# $ORACLE_FOLDER
# $DATA_FOLDER
# $EMB_FOLDER
# $PRETRAINED_EMBED
# $PRETRAINED_EMBED_DIM


##############################################################

###############################################################

##### decoding configuration
model_epoch=_last
# beam_size=1
batch_size=128

echo "[Cleaning pointer arcs and computing smatch:]"
for beam_size in 1 5 10
do
    . $dir/ad_test_post-decoding.sh "" dev
    . $dir/ad_test_post-decoding.sh "" test
done

cp $dir/ad_test_post-decoding.sh $MODEL_FOLDER/test_post-decoding.sh
