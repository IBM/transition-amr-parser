#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset

# Argument handling
# First argument must be checkpoint
HELP="\nbash $0 <checkpoint> [-o results_prefix] [-s (dev/test)] [-b beam_size]\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ ! -f "$1" ] && "Missing $1" && exit 1
checkpoint=$1
# process the rest with argument parser
results_prefix=""
data_split2=dev
beam_size=1
shift 
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) results_prefix="$2"; shift 2;;
    -s) data_split2="$2"; shift 2;;
    -b) beam_size="$2"; shift 2;;
    *) echo "unrecognized argument: $1"; exit 1;;
  esac
done

# activate virtualenenv and set other variables
. set_environment.sh

##### CONFIG
dir=$(dirname $0)
# if [ ! -z "${1+x}" ]; then
if [ ! -z "$1" ]; then
    config=$1
    . $config    # $config_data should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

# extract config from checkpoint path
model_folder=$(dirname $checkpoint)
config=$model_folder/config.sh
[ ! -f "$config" ] && "Missing $config" && exit 1

##### script specific config
if [ -z "$2" ]; then
    data_split_amr="dev"
else
    data_split_amr=$2
fi

# set data split parameters 
if [ $data_split2 == "dev" ]; then
    data_split=valid
    reference_amr=$AMR_DEV_FILE
    wiki=$WIKI_DEV
    reference_amr_wiki=$AMR_DEV_FILE_WIKI
elif [ $data_split2 == "test" ]; then
    data_split=test
    reference_amr=$AMR_TEST_FILE
    wiki=$WIKI_TEST
    reference_amr_wiki=$AMR_TEST_FILE_WIKI
else
    echo "$2 is invalid; must be dev or test"
fi

# model_epoch=${model_epoch:-_last}
# beam_size=${beam_size:-5}
# batch_size=${batch_size:-128}
# use_pred_rules=${use_pred_rules:-0}

RESULTS_FOLDER=$MODEL_FOLDER/beam${beam_size}
# Generate results_prefix name if not provided
if [ "$results_prefix" == "" ];then
    results_prefix=$RESULTS_FOLDER/${data_split}_$(basename $checkpoint)
fi

# TASK=${TASK:-amr_action_pointer}

##### DECODING
mkdir -p $RESULTS_FOLDER
# --nbest 3 \
# --quiet
python fairseq_ext/generate.py \
    $DATA_FOLDER  \
    --emb-dir $EMB_FOLDER \
    --user-dir ../fairseq_ext \
    --task $TASK \
    --gen-subset $data_split \
    --machine-type AMR  \
    --machine-rules $ORACLE_FOLDER/train.rules.json \
    --modify-arcact-score 1 \
    --use-pred-rules $USE_PRED_RULES \
    --beam $beam_size \
    --batch-size $BATCH_SIZE \
    --remove-bpe \
    --path $checkpoint \
    --quiet \
    --results-path $results_prefix \

# exit 0

##### Create the AMR from the model obtained actions
python transition_amr_parser/o8_fake_parse.py \
    --in-sentences $ORACLE_FOLDER/${data_split2}.en \
    --in-actions ${results_prefix}.actions \
    --out-amr ${results_prefix}.amr \
    --in-pred-entities $ENTITIES_WITH_PREDS \

# exit 0

##### SMATCH evaluation
if [[ "$wiki" == "" ]]; then

    # Smatch evaluation without wiki

    echo "Computing SMATCH ---"
    python smatch/smatch.py \
         --significant 4  \
         -f $reference_amr \
         $results_prefix.amr \
         -r 10 \
         > $results_prefix.smatch

    cat $results_prefix.smatch

else

    # Smatch evaluation with wiki

    # add wiki
    echo "Add wiki ---"
    python scripts/add_wiki.py \
        ${results_prefix}.amr $wiki \
        > ${results_prefix}.wiki.amr

    # compute score
    echo "Computing SMATCH ---"
    python smatch/smatch.py \
         --significant 4  \
         -f $reference_amr_wiki \
         ${results_prefix}.wiki.amr \
         -r 10 \
         > ${results_prefix}.wiki.smatch

    cat ${results_prefix}.wiki.smatch

fi
