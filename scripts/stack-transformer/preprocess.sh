set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

# Argument handling
config=$1

[ ! -f "$config" ] && echo "Missing $config" && exit 1

# Load config
. "$config"

# stage-1: Preprocess

# NORMALIZE AND ALIGN DATA (AMR only)
if [ "$TASK_TAG" == "AMR" ] && [ ! -f "$AMR_TRAIN_FILE" ];then

    # Dev
    python preprocess/remove_wiki.py $AMR_DEV_FILE_WIKI ${AMR_DEV_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_DEV_FILE_WIKI}.no_wiki $AMR_DEV_FILE
    
    # Test
    python preprocess/remove_wiki.py $AMR_TEST_FILE_WIKI ${AMR_TEST_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_TEST_FILE_WIKI}.no_wiki $AMR_TEST_FILE

    # Train
    python preprocess/remove_wiki.py $AMR_TRAIN_FILE_WIKI ${AMR_TRAIN_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_TRAIN_FILE_WIKI}.no_wiki $AMR_TRAIN_FILE

fi

# CREATE ORACLE DATA
[ ! -d $ORACLE_FOLDER ] && mkdir -p $ORACLE_FOLDER
if [ "$TASK_TAG" == "dep-parsing" ];then

    # nothing to do since the oracle is given, just copy it locally
    echo "cp $PTB_ORACLE/$ORACLE_TAG/* $ORACLE_FOLDER"
    cp $PTB_ORACLE/$ORACLE_TAG/* $ORACLE_FOLDER
    chmod u+w -R $ORACLE_FOLDER
    # dummy, will not be used
    entity_rules=""

elif [ "$TASK_TAG" == "AMR" ];then

    # Create entity rules if missing
    if [ ! -f "$ENTITY_RULES" ];then
        python scripts/extract_rules.py $AMR_TRAIN_FILE $ENTITY_RULES
    fi

    # compute oracles if missing
    if [ ! -f "$ORACLE_FOLDER/test.rules.json" ];then

        # Train
        amr-oracle \
            --in-amr $AMR_TRAIN_FILE \
            --entity-rules $ENTITY_RULES \
            --out-sentences $ORACLE_FOLDER/train.en \
            --out-actions $ORACLE_FOLDER/train.actions \
            --out-rule-stats $ORACLE_FOLDER/train.rules.json \
            $ORACLE_TRAIN_ARGS
        
        # Dev and test
        amr-oracle \
            --in-amr $AMR_DEV_FILE \
	        --entity-rules $ENTITY_RULES \
            --out-sentences $ORACLE_FOLDER/dev.en \
            --out-actions $ORACLE_FOLDER/dev.actions \
            --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
            $ORACLE_DEV_ARGS
    
        amr-oracle \
            --in-amr $AMR_TEST_FILE \
	        --entity-rules $ENTITY_RULES \
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
    python transition_amr_parser/mixer.py \
        --in-folders $AMR_ORACLE_FOLDER $NER_ORACLE_FOLDER \
        --in-tasks AMR NER \
        --out-folder $ORACLE_FOLDER \
        $ORACLE_MIXER_ARGS

else
    echo -e "Unknown task $TASK_TAG"
fi

# Dictionary update for fine-tuning. We will add the words from the fine-tuning
# vocabulary to the pretrained one. Note that there is a similar if in train.sh
# to adjust pretrained model embeddings accordingly
if [[ "$FAIRSEQ_TRAIN_ARGS" =~ .*"--restore-file".* ]];then

    # Work with a copy of the pretrained dictionaries (will be modified)
    mkdir -p $FEATURES_FOLDER

    # source 
    cp $PRETRAINED_SOURCE_DICT ${SRC_DICT}
    python scripts/create_fairseq_dicts.py \
        --in-pretrain-dict $SRC_DICT \
        --in-fine-tune-data $ORACLE_FOLDER/train.en \
    
    # target
    cp $PRETRAINED_TARGET_DICT ${TGT_DICT}
    python scripts/create_fairseq_dicts.py \
        --in-pretrain-dict $TGT_DICT \
        --in-fine-tune-data $ORACLE_FOLDER/train.actions \

fi

# PREPROCESSING
# extract data
echo "fairseq-preprocess $FAIRSEQ_PREPROCESS_ARGS"
fairseq-preprocess $FAIRSEQ_PREPROCESS_ARGS

# In fine-tune mode, we may need to adjust model size
if [[ "$FAIRSEQ_TRAIN_ARGS" =~ .*"--restore-file".* ]];then
    # We will modify the checkpoint, so we need to copy it
    [ ! -f "$RESTORE_FILE" ] && \
        cp $PRETRAINED_MODEL $RESTORE_FILE
    python scripts/merge_restored_vocabulary.py $FAIRSEQ_TRAIN_ARGS
fi
