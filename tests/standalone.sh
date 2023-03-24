set -o errexit
set -o pipefail
if [ -z $1 ];then

    # Standard mini-test with wiki25
    config=configs/wiki25-structured-bart-base-neur-al-sampling5.sh

else

    # custom config mini-test
    config=$1
fi
. set_environment.sh
set -o nounset

# load config
. $config

# use first seed
seed=$(echo $SEEDS | sed 's@ .*@@g')
# rest, from config
sset=test

reference_amr_wiki=$AMR_TEST_FILE_WIKI
wiki=$LINKER_CACHE_PATH/${sset}.wiki
checkpoint=${MODEL_FOLDER}seed${seed}/$DECODING_CHECKPOINT

# where to put results
FOLDER=${MODEL_FOLDER}seed${seed}/beam${BEAM_SIZE}/
results_prefix=$FOLDER/${sset}_$DECODING_CHECKPOINT

# needs data and model
[ ! -f "$checkpoint" ] \
    && echo "Missing $checkpoint" \
    && exit 1

# prepare unit test folder
[ -d "$FOLDER" ] && rm -R  $FOLDER/
mkdir -p $FOLDER

[ ! -f "$reference_amr_wiki" ] \
    && echo "Missing $reference_amr_wiki" \
    && exit 1

#    # extract sentences from test
#    grep '# ::tok ' $ALIGNED_FOLDER/${sset}.txt \
#        | sed 's@# ::tok @@g' > ${results_prefix}.tokens
#    
#    # run first seed of model
#    echo "amr-parse --beam ${BEAM_SIZE} --batch-size 128 -c $checkpoint -i ${results_prefix}.tokens -o ${results_prefix}.amr"
#    amr-parse --beam ${BEAM_SIZE} --batch-size 128 -c $checkpoint -i ${results_prefix}.tokens -o ${results_prefix}.amr

# extract sentences from test
grep '# ::snt ' $reference_amr_wiki \
    | sed 's@# ::snt @@g' > ${results_prefix}.sentences

# run first seed of model; --fp16 
cmd="amr-parse --beam ${BEAM_SIZE} --batch-size ${BATCH_SIZE} --tokenize -c $checkpoint -i ${results_prefix}.sentences -o ${results_prefix}.amr --out-tokens ${results_prefix}.tokens --out-actions ${results_prefix}.actions"
echo "$cmd"
eval "$cmd"
    
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
    echo "$reference_amr_wiki"
    echo "${results_prefix}.amr"
    smatch.py -r 10 --significant 4 \
         -f $reference_amr_wiki \
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
