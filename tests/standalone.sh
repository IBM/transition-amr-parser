set -o errexit
set -o pipefail
if [ -z $1 ];then

    # Standard mini-test with wiki25
    config=configs/wiki25-structured-bart-base-neur-al-sampling.sh

    # Delete previous runs if exists
    rm -Rf DATA/wiki25/*

    # replace code above with less restrictive deletion
    # rm -R -f DATA/wiki25/embeddings
    # rm -R -f DATA/wiki25/features
    # rm -R -f DATA/wiki25/oracles
    # rm -R -f DATA/wiki25/models

    # simulate completed corpora extraction and alignment
    bash tests/create_wiki25_mockup.sh

else

    # custom config mini-test
    config=$1
fi
set -o nounset

# load config
. $config

# from config
reference_amr=$AMR_DEV_FILE
reference_amr_wiki=$AMR_DEV_FILE_WIKI
wiki=$LINKER_CACHE_PATH/dev.wiki
checkpoint=${MODEL_FOLDER}-seed42/$DECODING_CHECKPOINT
# where to put results
FOLDER=${MODEL_FOLDER}-seed42/unit_test/
results_prefix=$FOLDER/dev
[ -d "$FOLDER" ] && rm $FOLDER/*
mkdir -p $FOLDER

[ ! -f "$ALIGNED_FOLDER/dev.txt" ] \
    && echo "Missing $ALIGNED_FOLDER/dev.txt" \
    && exit 1

# prepare model for export
# bash run/status.sh -c $config --final-remove

grep '^# ::tok' $ALIGNED_FOLDER/dev.txt \
    | sed 's@^# ::tok @@g' > ${results_prefix}.tokens

# run first seed of model
amr-parse -c $checkpoint -i ${results_prefix}.tokens -o ${results_prefix}.amr --beam 10 --batch-size 128

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

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.amr > ${results_prefix}.amr.no_isi

    echo "Computing SMATCH between ---"
    echo "$reference_amr"
    echo "${results_prefix}.amr"
    smatch.py -r 10 --significant 4 \
         -f $reference_amr \
         ${results_prefix}.amr.no_isi \
         | tee ${results_prefix}.smatch

elif [[ "$EVAL_METRIC" == "wiki.smatch" ]]; then

    # Smatch evaluation without wiki

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.wiki.amr > ${results_prefix}.wiki.amr.no_isi

    # compute score
    echo "Computing SMATCH between ---"
    echo "$reference_amr_wiki"
    echo "${results_prefix}.wiki.amr"
    smatch.py -r 10 --significant 4  \
         -f $reference_amr_wiki \
         ${results_prefix}.wiki.amr.no_isi \
         | tee ${results_prefix}.wiki.smatch

fi
