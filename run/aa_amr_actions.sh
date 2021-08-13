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

    if [ ! -f "$ALIGN_MODEL" ];then

        # need to train a model (will include pre-procesing)
        # TODO: Add if [ JAMR_EVAL ] here to ignore this
        bash run/train_aligner.sh $config

    else

        # pre-procesing

        # remove wiki
        python preprocess/remove_wiki.py $AMR_TRAIN_FILE_WIKI ${AMR_TRAIN_FILE_WIKI}.no_wiki
        python preprocess/remove_wiki.py $AMR_DEV_FILE_WIKI ${AMR_DEV_FILE_WIKI}.no_wiki
        python preprocess/remove_wiki.py $AMR_TEST_FILE_WIKI ${AMR_TEST_FILE_WIKI}.no_wiki

        # tokenize sentences unless we use JAMR reference
        # use --simple for barebones tokenization otherwise imitates JAMR
        python scripts/tokenize_amr.py --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki --simple
        python scripts/tokenize_amr.py --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki --simple
        python scripts/tokenize_amr.py --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki --simple
    
        # generate ELMO vocabulary
        python align_cfg/vocab.py \
            --in-amrs ${AMR_TRAIN_FILE_WIKI}.no_wiki \
            --out-text $ALIGN_VOCAB_TEXT \
            --out-amr $ALIGN_VOCAB_AMR
    
        # Generate embeddings for the aligner
        python align_cfg/pretrained_embeddings.py --cuda \
            --cache-dir $ALIGNED_FOLDER \
            --vocab $ALIGN_VOCAB_TEXT
        python align_cfg/pretrained_embeddings.py --cuda \
            --cache-dir $ALIGNED_FOLDER \
            --vocab $ALIGN_VOCAB_AMR

    fi

    # Train
    echo "align train"
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGN_MODEL \
        --load-flags $ALIGN_MODEL_FLAGS \
        --vocab-text $ALIGN_VOCAB_TEXT \
        --vocab-amr $ALIGN_VOCAB_AMR \
        --write-single \
        --single-input ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --single-output $ALIGNED_FOLDER/train.txt

    # Dev
    echo "align dev"
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGN_MODEL \
        --load-flags $ALIGN_MODEL_FLAGS \
        --vocab-text $ALIGN_VOCAB_TEXT \
        --vocab-amr $ALIGN_VOCAB_AMR \
        --write-single \
        --single-input ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --single-output $ALIGNED_FOLDER/dev.txt
    
    # Test
    echo "align test"
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGN_MODEL \
        --load-flags $ALIGN_MODEL_FLAGS \
        --vocab-text $ALIGN_VOCAB_TEXT \
        --vocab-amr $ALIGN_VOCAB_AMR \
        --write-single \
        --single-input ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --single-output $ALIGNED_FOLDER/test.txt

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

    echo -e "\nTrain data"
   
    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TRAIN_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-tokens $ORACLE_FOLDER/train.en \
        --absolute-stack-positions  \
        --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    echo -e "\nDev data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_DEV_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --out-tokens $ORACLE_FOLDER/dev.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    echo -e "\nTest data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TEST_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/test.actions \
        --out-tokens $ORACLE_FOLDER/test.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    touch $ORACLE_FOLDER/.done

fi
