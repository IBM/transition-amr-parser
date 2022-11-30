set -o errexit
set -o pipefail
if [ -z $1 ];then

    # Standard mini-test with wiki25
    config=configs/both_doc+sen_trainsliding_ws400x100.sh

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

reference_amr=$ORACLE_FOLDER/${sset}_docAMR.docamr
# wiki=$LINKER_CACHE_PATH/${sset}.wiki
checkpoint=${MODEL_FOLDER}seed${seed}/$DECODING_CHECKPOINT

force_actions=$ORACLE_FOLDER/${sset}.force_actions
# where to put results
FOLDER=${MODEL_FOLDER}seed${seed}/beam${BEAM_SIZE}_aftermerge/
results_prefix=$FOLDER/${sset}_$DECODING_CHECKPOINT

# needs data and model
[ ! -f "$checkpoint" ] \
    && echo "Missing $checkpoint" \
    && exit 1

# prepare unit test folder
# [ -d "$FOLDER" ] && rm -R  $FOLDER/
mkdir -p $FOLDER

[ ! -f "$reference_amr" ] \
    && echo "Missing $reference_amr" \
    && exit 1

[ ! -f "$force_actions" ] \
    && echo "Missing $force_actions" \
    && force_actions=""

#    # extract sentences from test
#    grep '# ::tok ' $ALIGNED_FOLDER/${sset}.txt \
#        | sed 's@# ::tok @@g' > ${results_prefix}.tokens
#    
#    # run first seed of model
#    echo "amr-parse --beam ${BEAM_SIZE} --batch-size 128 -c $checkpoint -i ${results_prefix}.tokens -o ${results_prefix}.amr"
#    amr-parse --beam ${BEAM_SIZE} --batch-size 128 -c $checkpoint -i ${results_prefix}.tokens -o ${results_prefix}.amr

# extract sentences from test
# grep '# ::tok ' $reference_amr \
#     | sed 's@# ::tok @@g' > ${results_prefix}.sentences
cp $ORACLE_FOLDER/${sset}.en ${results_prefix}.sentences
sed -e 's/[[:space:]]\+/ /g' ${results_prefix}.sentences > ${results_prefix}.sentences_notab
# run first seed of model
cmd="amr-parse --fp16 --beam ${BEAM_SIZE} --batch-size ${BATCH_SIZE} -c $checkpoint -i ${results_prefix}.sentences_notab -o ${results_prefix}.amr --sliding --window-size 400 --window-overlap 100 --in-actions $force_actions"
echo "$cmd"
eval "$cmd"
    
# GRAPH POST-PROCESSING

# if [ "$LINKER_CACHE_PATH" == "" ];then

#     # just copy AMR to wiki AMR
#     cp ${results_prefix}.amr ${results_prefix}.wiki.amr

# # TODO: Unelegant detection of linker method (temporary)
# elif [ -f "${LINKER_CACHE_PATH}/trn.wikis" ];then

#     # Legacy linker 
#     python scripts/add_wiki.py \
#         ${results_prefix}.amr $wiki $LINKER_CACHE_PATH \
#         > ${results_prefix}.wiki.amr

# else

#     # BLINK cache
#     python scripts/retyper.py \
#         --inputfile ${results_prefix}.amr \
#         --outputfile ${results_prefix}.wiki.amr \
#         --skipretyper \
#         --wikify \
#         --blinkcachepath $LINKER_CACHE_PATH \
#         --blinkthreshold 0.0

# fi

## Change rep of docamr to docAMR for smatch
    
echo -e "\n Changing rep of dev/test data to docAMR "
doc-amr \
    --in-doc-amr-pairwise ${results_prefix}.amr \
    --rep docAMR \
    --pairwise-coref-rel same-as \
    --out-amr ${results_prefix}_docAMR.amr
results_prefix=${results_prefix}_docAMR

##### SMATCH evaluation
if [[ "$EVAL_METRIC" == "smatch" ]]; then

    # Smatch evaluation without wiki

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.amr > ${results_prefix}.amr.no_isi

    echo "Computing SMATCH between ---"
    echo "$reference_amr"
    echo "${results_prefix}.amr"
    doc-smatch -r 1 --significant 4 \
         -f $reference_amr \
         ${results_prefix}.amr.no_isi \
         | tee ${results_prefix}.smatch

elif [[ "$EVAL_METRIC" == "wiki.smatch" ]]; then

    # Smatch evaluation without wiki

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.wiki.amr > ${results_prefix}.wiki.amr.no_isi

    # compute score
    echo "Computing SMATCH between ---"
    echo "$reference_amr"
    echo "${results_prefix}.wiki.amr"
    doc-smatch -r 1 --significant 4  \
         -f $reference_amr \
         ${results_prefix}.wiki.amr.no_isi \
         | tee ${results_prefix}.wiki.smatch

fi
