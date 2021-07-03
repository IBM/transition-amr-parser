#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
config=$1
[ ! -f "$config" ] && "Missing $config" && exit 1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# We will need this to save the alignment log
mkdir -p $ORACLE_FOLDER

###### AMR Alignment
if [ -f $ALIGNED_FOLDER/.done ]; then

    echo "Directory to aligner: $ALIGNED_FOLDER already exists --- do nothing."

else

    mkdir -p $ALIGNED_FOLDER

    # Train
    python preprocess/remove_wiki.py $AMR_TRAIN_FILE_WIKI ${AMR_TRAIN_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_TRAIN_FILE_WIKI}.no_wiki $ALIGNED_FOLDER/train.txt

    # Dev
    python preprocess/remove_wiki.py $AMR_DEV_FILE_WIKI ${AMR_DEV_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_DEV_FILE_WIKI}.no_wiki $ALIGNED_FOLDER/dev.txt
    
    # Test
    python preprocess/remove_wiki.py $AMR_TEST_FILE_WIKI ${AMR_TEST_FILE_WIKI}.no_wiki
    bash preprocess/align.sh ${AMR_TEST_FILE_WIKI}.no_wiki $ALIGNED_FOLDER/test.txt

    # Mark as done
    touch $ALIGNED_FOLDER/.done

fi


##### ORACLE EXTRACTION
# Given sentence and aligned AMR, provide action sequence that generates the AMR back
if [ -f $ORACLE_FOLDER/.done ]; then
    
    echo "Directory to oracle: $ORACLE_FOLDER already exists --- do nothing."

else

    mkdir -p $ORACLE_FOLDER

    # copy the original AMR data: no wikification
    cp $ALIGNED_FOLDER/train.txt $ORACLE_FOLDER/ref_train.amr
    cp $ALIGNED_FOLDER/dev.txt $ORACLE_FOLDER/ref_dev.amr
    cp $ALIGNED_FOLDER/test.txt $ORACLE_FOLDER/ref_test.amr
   
    python transition_amr_parser/amr_oracle.py \
        --in-amr $AMR_TRAIN_FILE \
        --entity-rules $ORACLE_FOLDER/entity_rules.json \
        --out-sentences $ORACLE_FOLDER/train.en \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-rule-stats $ORACLE_FOLDER/train.rules.json \
        --copy-lemma-action
    
    python transition_amr_parser/amr_oracle.py \
        --in-amr $AMR_DEV_FILE \
        --entity-rules $ORACLE_FOLDER/entity_rules.json \
        --out-sentences $ORACLE_FOLDER/dev.en \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
        --copy-lemma-action
    
    python transition_amr_parser/amr_oracle.py \
        --in-amr $AMR_TEST_FILE \
        --entity-rules $ORACLE_FOLDER/entity_rules.json \
        --out-sentences $ORACLE_FOLDER/test.en \
        --out-actions $ORACLE_FOLDER/test.actions \
        --out-rule-stats $ORACLE_FOLDER/test.rules.json \
        --copy-lemma-action
    
    # Mark as done
    touch $ORACLE_FOLDER/.done

fi
