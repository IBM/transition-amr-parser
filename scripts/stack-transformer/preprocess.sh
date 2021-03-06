set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

# Argument handling
config=$1

# Load config
. "$config"

# stage-1: Preprocess

# ORACLE
[ ! -d $ORACLE_FOLDER ] && mkdir -p $ORACLE_FOLDER


# Create oracle data
if [ "$TASK_TAG" == "dep-parsing" ];then

    # nothing to do since the oracle is given, just copy it locally
    echo "cp $PTB_ORACLE/$ORACLE_TAG/* $ORACLE_FOLDER"
    cp $PTB_ORACLE/$ORACLE_TAG/* $ORACLE_FOLDER
    chmod u+w -R $ORACLE_FOLDER

elif [ "$TASK_TAG" == "AMR" ];then

    # Use custom entity rules or create them
    if [ -n "${ENTITY_RULES:-}" ] && [ "${ENTITY_RULES}" != "" ]; then
        entity_rules=$ENTITY_RULES

        # Exit with error if they do not exist
        [ ! -f "$ENTITY_RULES" ] && echo "Missing $ENTITY_RULES" & exit 1

    else
        entity_rules=$ORACLE_FOLDER/entity_rules.json
        if [ ! -f "$entity_rules" ];then
            python scripts/extract_rules.py \
                $AMR_TRAIN_FILE \
                $ORACLE_FOLDER/entity_rules.json
        fi
    fi

    if [ ! -f "$ORACLE_FOLDER/test.rules.json" ];then

        # Train
        amr-oracle \
            --in-amr $AMR_TRAIN_FILE \
            --entity-rules $entity_rules \
            --out-sentences $ORACLE_FOLDER/train.en \
            --out-actions $ORACLE_FOLDER/train.actions \
            --out-rule-stats $ORACLE_FOLDER/train.rules.json \
            $ORACLE_TRAIN_ARGS
        
        # Dev and test
        amr-oracle \
            --in-amr $AMR_DEV_FILE \
	        --entity-rules $entity_rules \
            --out-sentences $ORACLE_FOLDER/dev.en \
            --out-actions $ORACLE_FOLDER/dev.actions \
            --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
            $ORACLE_DEV_ARGS
    
        amr-oracle \
            --in-amr $AMR_TEST_FILE \
	        --entity-rules $entity_rules \
            --out-sentences $ORACLE_FOLDER/test.en \
            --out-actions $ORACLE_FOLDER/test.actions \
            --out-rule-stats $ORACLE_FOLDER/test.rules.json \
            $ORACLE_DEV_ARGS
    
    fi
    
elif [ "$TASK_TAG" == "NER" ];then

    # Concatenate all data into one file
    mkdir -p $(dirname $NER_TRAIN_FILE)
    cat $KLUE3_FOLDER/*_train.dat > $NER_TRAIN_FILE
    cat $KLUE3_FOLDER/*_devtest.dat > $NER_DEV_FILE
    cat $KLUE3_FOLDER/*_test.dat > $NER_TEST_FILE

    # train
    python bio_tags/oracle.py \
        --in-annotations $NER_TRAIN_FILE \
        --out-tokens $ORACLE_FOLDER/train.en \
        --out-actions $ORACLE_FOLDER/train.actions \
        --crop-tags \
        $ORACLE_TRAIN_ARGS

    # dev test
    python bio_tags/oracle.py \
        --in-annotations $NER_DEV_FILE \
        --out-tokens $ORACLE_FOLDER/dev.en \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --crop-tags \
        $ORACLE_DEV_ARGS

    # dev test
    python bio_tags/oracle.py \
        --in-annotations $NER_TEST_FILE \
        --out-tokens $ORACLE_FOLDER/test.en \
        --out-actions $ORACLE_FOLDER/test.actions \
        --crop-tags \
        $ORACLE_DEV_ARGS

elif [ "$TASK_TAG" == "NER+AMR" ];then
    
    # sanity check: oracles extracted for the two tasks
    [ ! -d $AMR_ORACLE_FOLDER ] && echo "Missing oracle $AMR_ORACLE_FOLDER" && exit 1
    [ ! -d $NER_ORACLE_FOLDER ] && echo "Missing oracle $NER_ORACLE_FOLDER" && exit 1
    
    # oracle actions 
    # This is only avaliable in modular_semantic_parser/
    python mixer.py \
        --in-folders $AMR_ORACLE_FOLDER $NER_ORACLE_FOLDER \
        --in-tasks AMR NER \
        --out-folder $ORACLE_FOLDER \
        $ORACLE_MIXER_ARGS
            
else
    echo -e "Unknown task $TASK"
fi

# PREPROCESSING
# extract data
echo "fairseq-preprocess $FAIRSEQ_PREPROCESS_ARGS"
fairseq-preprocess --entity-rules $entity_rules $FAIRSEQ_PREPROCESS_ARGS
