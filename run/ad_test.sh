#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
# First argument must be checkpoint
HELP="\nbash $0 <checkpoint> [-o results_prefix] [-s (dev/test)] [-b beam_size]\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
first_path=$(echo $1 | sed 's@:.*@@g')
[ ! -f "$first_path" ] && "Missing $1" && exit 1
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

set -o nounset
# extract config from checkpoint path
model_folder=$(dirname $first_path)
config=$model_folder/config.sh
[ ! -f "$config" ] && "Missing $config" && exit 1

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# set data split parameters 
if [ $data_split2 == "dev" ]; then
    data_split=valid
    data_split_name=dev
    reference_amr=$AMR_DEV_FILE
    wiki=$LINKER_CACHE_PATH/dev.wiki
    reference_amr_wiki=$AMR_DEV_FILE_WIKI
elif [ $data_split2 == "test" ]; then
    data_split=test
    data_split_name=test
    reference_amr=$AMR_TEST_FILE
    wiki=$LINKER_CACHE_PATH/test.wiki
    reference_amr_wiki=$AMR_TEST_FILE_WIKI
else
    echo "$2 is invalid; must be dev or test"
    exit 1
fi

# we may have to re-compute features
if [[ (-f $DATA_FOLDER/.done) && (-f $EMB_FOLDER/.done) ]]; then
    echo "Using $DATA_FOLDER/"
else
    echo "Re-computing features, may take a while"
    bash run/ab_preprocess.sh $config
fi

RESULTS_FOLDER=$(dirname $first_path)/beam${beam_size}
# Generate results_prefix name if not provided
if [ "$results_prefix" == "" ];then
    results_prefix=$RESULTS_FOLDER/${data_split}_$(basename $first_path)
fi
echo "Generating ${results_prefix}.actions"

##### DECODING
mkdir -p $RESULTS_FOLDER
# --nbest 3 \
# --quiet

if [ ! -f "${results_prefix}.actions" ];then

    python fairseq_ext/generate.py \
        $DATA_FOLDER  \
        --emb-dir $EMB_FOLDER \
        --user-dir ./fairseq_ext \
        --task $TASK \
        --gen-subset $data_split \
        --src-fix-emb-use $src_fix_emb_use \
        --machine-type AMR  \
        --machine-rules $ORACLE_FOLDER/train.rules.json \
        --machine-config $ORACLE_FOLDER/machine_config.json \
        --modify-arcact-score 1 \
        --use-pred-rules $USE_PRED_RULES \
        --beam $beam_size \
        --batch-size $BATCH_SIZE \
        --remove-bpe \
        --path $checkpoint \
        --quiet \
        --results-path $results_prefix 

fi

##### Create the AMR from the model obtained actions
python transition_amr_parser/amr_machine.py \
    --in-machine-config $ORACLE_FOLDER/machine_config.json \
    --in-tokens $ORACLE_FOLDER/${data_split_name}.en \
    --in-actions ${results_prefix}.actions \
    --out-amr ${results_prefix}.amr


# GRAPH POST-PROCESSING

if [ "$LINKER_CACHE_PATH" == "" ];then

    # just copy AMR to wiki AMR
    cp ${results_prefix}.amr ${results_prefix}.wiki.amr

# TODO: Unelegant detection of linker method (temporary)
elif [ -f "${LINKER_CACHE_PATH}/trn.wikis" ];then

    # Legacy linker 
    python scripts/add_wiki.py \
        ${results_prefix}.amr $wiki $LINKER_CACHE_PATH \
        > ${results_prefix}.wiki.amr

else

    # BLINK cache
    python scripts/retyper.py \
        --inputfile ${results_prefix}.amr \
        --outputfile ${results_prefix}.wiki.amr \
        --skipretyper \
        --wikify \
        --blinkcachepath $LINKER_CACHE_PATH \
        --blinkthreshold 0.0

fi


##### SMATCH evaluation
if [[ "$EVAL_METRIC" == "smatch" ]]; then

    # Smatch evaluation without wiki

    echo "Computing SMATCH between ---"
    echo "$reference_amr"
    echo "${results_prefix}.amr"
    smatch.py \
         --significant 4  \
         -f $reference_amr \
         ${results_prefix}.amr \
         -r 10 \
         | tee ${results_prefix}.smatch

elif [[ "$EVAL_METRIC" == "wiki.smatch" ]]; then

    # compute score
    echo "Computing SMATCH between ---"
    echo "$reference_amr_wiki"
    echo "${results_prefix}.wiki.amr"
    smatch.py \
         --significant 4  \
         -f $reference_amr_wiki \
         ${results_prefix}.wiki.amr \
         -r 10 \
         | tee ${results_prefix}.wiki.smatch

fi
