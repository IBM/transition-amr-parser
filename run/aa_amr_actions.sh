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

# Require aligned AMR
[ ! -f "$ALIGNED_FOLDER/.done" ] && \
    echo -e "\nRequires amr alignment in $ALIGNED_FOLDER\n" && \
    exit 1

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
    
    if [[ ! "$WIKI_DEV" == "" ]]; then
        # copy the original AMR data: wiki files and original AMR with wikification
        cp $WIKI_DEV $ORACLE_FOLDER/ref_dev.wiki
        cp $WIKI_TEST $ORACLE_FOLDER/ref_test.wiki
        cp $AMR_DEV_FILE_WIKI $ORACLE_FOLDER/ref_dev.wiki.amr
        cp $AMR_TEST_FILE_WIKI $ORACLE_FOLDER/ref_test.wiki.amr
    fi

    # generate the actions
    
    if [[ $MAX_WORDS == 100 ]]; then
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_TRAIN_FILE \
    	    --in-pred-entities $ENTITIES_WITH_PREDS \
            --out-sentences $ORACLE_FOLDER/train.en \
            --out-actions $ORACLE_FOLDER/train.actions \
            --out-rule-stats $ORACLE_FOLDER/train.rules.json \
            --multitask-max-words $MAX_WORDS  \
            --out-multitask-words $ORACLE_FOLDER/train.multitask_words \
            --copy-lemma-action
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_DEV_FILE \
    	    --in-pred-entities $ENTITIES_WITH_PREDS\
            --out-sentences $ORACLE_FOLDER/dev.en \
            --out-actions $ORACLE_FOLDER/dev.actions \
            --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
            --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
            --copy-lemma-action
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_TEST_FILE \
    	    --in-pred-entities $ENTITIES_WITH_PREDS\
            --out-sentences $ORACLE_FOLDER/test.en \
            --out-actions $ORACLE_FOLDER/test.actions \
            --out-rule-stats $ORACLE_FOLDER/test.rules.json \
            --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
            --copy-lemma-action
    
    elif [[ $MAX_WORDS == 0 ]]; then
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_TRAIN_FILE \
    	    --in-pred-entities $ENTITIES_WITH_PREDS \
            --out-sentences $ORACLE_FOLDER/train.en \
            --out-actions $ORACLE_FOLDER/train.actions \
            --out-rule-stats $ORACLE_FOLDER/train.rules.json \
            --copy-lemma-action
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_DEV_FILE \
    	    --in-pred-entities $ENTITIES_WITH_PREDS\
            --out-sentences $ORACLE_FOLDER/dev.en \
            --out-actions $ORACLE_FOLDER/dev.actions \
            --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
            --copy-lemma-action
    
        python transition_amr_parser/amr_oracle.py \
            --in-amr $AMR_TEST_FILE \
        	--in-pred-entities $ENTITIES_WITH_PREDS\
            --out-sentences $ORACLE_FOLDER/test.en \
            --out-actions $ORACLE_FOLDER/test.actions \
            --out-rule-stats $ORACLE_FOLDER/test.rules.json \
            --copy-lemma-action
    
    else
    
        echo "MAX_WORDS ${MAX_WORDS} not allowed" && exit 1
    
    fi

    # Mark as done
    touch $ORACLE_FOLDER/.done

fi
