#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

##### CONFIG
dir=$(dirname $0)
if [ -z "$1" ]; then
    config="config.sh"
else
    config=$1
fi
. $dir/$config    # we should always call from one level up

##### script specific config
if [ -z "$2" ]; then
    data_split_amr="dev"
else
    data_split_amr=$2
fi

if [ $data_split_amr == "dev" ]; then
    data_split=valid
    reference_amr=$AMR_DEV_FILE
elif [ $data_split_amr == "test" ]; then
    data_split=test
    reference_amr=$AMR_TEST_FILE
else
    echo "$2 is invalid; must be dev or test"
fi
    

# data_split=valid
# data_split_amr=dev    # TODO make the names consistent
# reference_amr=$AMR_DEV_FILE

# data_split=test
# data_split_amr=test    # TODO make the names consistent
# reference_amr=$AMR_TEST_FILE

model_epoch=_last
beam_size=10

RESULTS_FOLDER=$MODEL_FOLDER/beam${beam_size}
results_prefix=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}.nopos-score
model=$MODEL_FOLDER/checkpoint${model_epoch}.pt


##### DECODING
# rm -Rf $RESULTS_FOLDER
mkdir -p $RESULTS_FOLDER
# --nbest 3 \
# --quiet
python fairseq_ext/generate.py \
    $DATA_FOLDER  \
    --user-dir ../fairseq_ext \
    --task amr_pointer \
    --gen-subset $data_split \
    --machine-type AMR  \
    --machine-rules $ORACLE_FOLDER/train.rules.json \
    --beam $beam_size \
    --batch-size 128 \
    --remove-bpe \
    --path $model  \
    --quiet \
    --results-path $results_prefix \

# exit 0
    
##### Create the AMR from the model obtained actions
python transition_amr_parser/o7_fake_parse.py \
    --in-sentences $ORACLE_FOLDER/$data_split_amr.en \
    --in-actions $results_prefix.actions \
    --out-amr $results_prefix.amr \

# exit 0

#### Smatch evaluation without wiki
python smatch/smatch.py \
     --significant 4  \
     -f $reference_amr \
     $results_prefix.amr \
     -r 10 \
     > $results_prefix.smatch

cat $results_prefix.smatch
