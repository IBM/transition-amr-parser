set -o pipefail
set -o errexit
# setup environment
. set_environment.sh
[ "$#" -lt 1 ] && \
    echo -e "\nepoch_tester.sh <model_folder_1> [<model_folder_2> ...]\n" && \
    exit 1
set -o nounset

# Loop over one or more model folders (where checkpoints are)
for model_folder in "$@";do

    # a model folder should have a config.sh
    config="$model_folder/config.sh"
    [ ! -f "$config" ] && \
        echo "Missing $config" && \
        exit 1

    # FIXME: Each config should be load in a separate subshell (bash function?)
    # activate config
    . "$config"

    # we will store results here 
    output_folder="$model_folder/epoch_tests/"
    mkdir -p "$output_folder"

    # Loop over existing checkpoints
    for test_model in $(find $model_folder -iname 'checkpoint[0-9]*.pt' | sort -r);do
    
        # basename for all experiments of this checkpoint 
        std_name=$output_folder/dec-$(basename $test_model .pt)
    
        # FIXME: Explicit ENTITY_RULES handling in config
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
    
        # Evaluation
        if [ "$TASK_TAG" == "dep-parsing" ];then

            echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}"
            fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}
        
            # Create the AMR from the model obtained actions
            python scripts/dep_parsing_score.py \
                --in-tokens $ORACLE_FOLDER/dev.en \
                --in-actions ${std_name}.actions \
                --in-gold-actions $ORACLE_FOLDER/dev.actions  \
                > ${std_name}.las
            cat ${std_name}.las
    
        elif [ "$TASK_TAG" == "AMR" ];then
    
            echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name} --entity-rules $ENTITY_RULES"
            fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name} --entity-rules $ENTITY_RULES

            # TODO: Path fixed at the config, create if it does not exist

    	    if [ "$ENTITY_RULES" == "" ]; then
    	        ENTITY_RULES=$ORACLE_FOLDER/entity_rules.json
    	    fi
    
            # bear in mind this has hardcoded dev here 
            amr-fake-parse \
    	        --entity-rules $ENTITY_RULES \
                --in-sentences $ORACLE_FOLDER/dev.en \
                --in-actions ${std_name}.actions \
                --out-amr ${std_name}.amr \
                --sanity-check
    
            if [ "$WIKI_DEV" == "" ];then
    
                # Smatch evaluation without wiki
                python smatch/smatch.py \
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
            
                python smatch/smatch.py \
                     --significant 4  \
                     -f $AMR_DEV_FILE_WIKI \
                     ${std_name}.wiki.amr \
                     -r 10 \
                     > ${std_name}.wiki.smatch
                
                # plot score
                cat ${std_name}.wiki.smatch
        
            fi
        
        elif [ "$TASK_TAG" == "NER" ];then

            echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}"
            fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}
        
            # play actions to create annotations
            python transition_amr_parser/play.py \
                --in-tokens $ORACLE_FOLDER/dev.en \
                --in-actions ${std_name}.actions \
                --machine-type NER \
                --out-annotations-folder $(dirname ${std_name}) \
                --basename $(basename ${std_name}) \
            
            # measure performance
            python transition_amr_parser/metrics.py \
                --in-annotations ${std_name}.dat \
                --in-reference-annotations $NER_DEV_FILE \
                --out-score ${std_name}.f-measure
            cat ${std_name}.f-measure
        
        elif [ "$TASK_TAG" == "NER+AMR" ];then
        
            # AMR scores
            python transition_amr_parser/play.py \
                --in-tokens $ORACLE_FOLDER/dev.en \
                --in-actions ${std_name}.actions \
                --in-mixing-indices $ORACLE_FOLDER/dev.mixing_indices \
                --out-annotations-folder $(dirname ${std_name}) \
                --basename $(basename ${std_name}) \
            
            # compute F-measure for NER
            python transition_amr_parser/metrics.py \
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
    
        # clean-up all checkpoints and save the *_best_* labeled ones
        # bash scripts/stack-transformer/remove_checkpoints.sh $model_folder
    
    fi
 

done
