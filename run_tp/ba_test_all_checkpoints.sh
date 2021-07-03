#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\ntest_all_checkpoints.sh <checkpoints_folder>\n" && \
    exit 1
set -o nounset


# Get the data confirguration
# TODO copy the latest data configuration into the data config (done)
expdir=$(dirname $checkpoints_folder)
config_data=($expdir/config_data_*)
echo -e "\n[Experiments directory:]"
echo $expdir
echo -e "\n[Data configuration file:]"
echo $config_data
. $config_data
# now we have
# echo $AMR_TRAIN_FILE
# echo $AMR_DEV_FILE
# echo $AMR_TEST_FILE
# echo $WIKI_DEV
# echo $AMR_DEV_FILE_WIKI
# echo $WIKI_TEST
# echo $AMR_TEST_FILE_WIKI
#
# echo $DATA_FOLDER
# echo $ORACLE_FOLDER
# echo $EMB_FOLDER


# Loop over existing checkpoints
# NOTE sort -r doesn't really sort the epoch numbers now; 9* is larger than any 1* by default
for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r); do

    echo -e "\n$test_model"
    echo "[Decoding and computing smatch:]"
    
    MODEL_FOLDER=$checkpoints_folder
    # or: MODEL_FOLDER=$(dirname $test_model)
    
    # decoding setup
    model_epoch=$(basename $test_model | sed 's/checkpoint\(.*\).pt/\1/g')
    beam_size=1
    batch_size=128
    use_pred_rules=0
    
    # run decoding
    . run_tp/ad_test.sh "" dev
    
done

touch $checkpoints_folder/model-selection_stage1-done

exit 0





for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r); do

    # pytorch model folder and basename for this checkpoints data
    model_folder=$(dirname $test_model)
    output_folder="$model_folder/epoch_tests/"
    config="$model_folder/config.sh"
    std_name=$output_folder/dec-$(basename $test_model .pt)

    # logs for each run of the checkpoint will be stored here
    mkdir -p "$output_folder"

    # activate config
    . "$config"

    if [ "$TASK_TAG" == "AMR" ] ; then
	if [ -n "${ENTITY_RULES:-}" ] && [ -f "$ENTITY_RULES" ] ; then
            echo "using given entity rules"
	else
            echo "reading entity rules from oracle"
            ENTITY_RULES=$ORACLE_FOLDER/entity_rules.json
	fi
    fi


    # skip if decoding ran already once
    if [ -f "${std_name}.actions" ];then
        echo -e "Skipping $std_name"
        continue
    fi

    # decode 
    if [ ! -f "$test_model" ];then
        # model was meanwhile deleted
        continue
    fi

    echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}"
    fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name} --entity-rules $ENTITY_RULES
    
    # Create oracle data
    # Create oracle data
    if [ "$TASK_TAG" == "dep-parsing" ];then
    
        # Create the AMR from the model obtained actions
        python scripts/dep_parsing_score.py \
            --in-tokens $ORACLE_FOLDER/dev.en \
            --in-actions ${std_name}.actions \
            --in-gold-actions $ORACLE_FOLDER/dev.actions  \
            > ${std_name}.las
        cat ${std_name}.las

    elif [ "$TASK_TAG" == "AMR" ];then

        # TODO: Path fixed at the config, create if it does not exist
	    if [ "$ENTITY_RULES" == "" ]; then
	        ENTITY_RULES=$ORACLE_FOLDER/entity_rules.json
	    fi

        # bear in mind this has hardcoded dev here 
        amr-fake-parse \
	    --entity-rules $ENTITY_RULES \
            --in-sentences $ORACLE_FOLDER/dev.en \
            --in-actions ${std_name}.actions \
            --out-amr ${std_name}.amr 

        if [ "$WIKI_DEV" == "" ];then

            # Smatch evaluation without wiki
            smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE \
                 ${std_name}.amr \
                 -r 10 \
                 > ${std_name}.smatch
            
            # plot score
            cat ${std_name}.smatch

        else

            # Smatch evaluation with wiki
            # add wiki
            python scripts/add_wiki.py \
                ${std_name}.amr $WIKI_DEV \
                > ${std_name}.wiki.amr
        
            smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE_WIKI \
                 ${std_name}.wiki.amr \
                 -r 10 \
                 > ${std_name}.wiki.smatch
            
            # plot score
            cat ${std_name}.wiki.smatch
    
        fi
    
    elif [ "$TASK_TAG" == "NER" ];then
    
        # play actions to create annotations
        python play.py \
            --in-tokens $ORACLE_FOLDER/dev.en \
            --in-actions ${std_name}.actions \
            --machine-type NER \
            --out-annotations-folder $(dirname ${std_name}) \
            --basename $(basename ${std_name}) \
        
        # measure performance
        python bio_tags/metrics.py \
            --in-annotations ${std_name}.dat \
            --in-reference-annotations $NER_DEV_FILE \
            --out-score ${std_name}.f-measure
        cat ${std_name}.f-measure
    
    elif [ "$TASK_TAG" == "NER+AMR" ];then
    
        # AMR scores
        python play.py \
            --in-tokens $ORACLE_FOLDER/dev.en \
            --in-actions ${std_name}.actions \
            --in-mixing-indices $ORACLE_FOLDER/dev.mixing_indices \
            --out-annotations-folder $(dirname ${std_name}) \
            --basename $(basename ${std_name}) \
        
        # compute F-measure for NER
        python bio_tags/metrics.py \
            --in-annotations ${std_name}.dat \
            --in-reference-annotations $NER_DEV_FILE \
            --out-score ${std_name}.f-measure
        cat ${std_name}/dev.f-measure
        
        # compute smatch for AMR
        smatch.py \
             --significant 4  \
             -f $AMR_DEV_FILE \
             ${std_name}.amr \
             -r 10 \
             > ${std_name}.smatch
        cat ${std_name}.smatch
    
    fi
    
done

# After all tests are done, rank model and softlink the top 3 models according
# to smatch
if [ "$TASK_TAG" == "AMR" ];then

    # model linking (will also display table)
    python scripts/stack-transformer/rank_model.py --link-best

fi