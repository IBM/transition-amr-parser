#!/bin/bash

set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset


##### CONFIG
dir=$(dirname $0)
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config=$1
    . $config    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

##### script specific config


##### ORACLE EXTRACTION
# Given sentence and aligned AMR, provide action sequence that generates the AMR back
# [ -d $ORACLE_FOLDER ] && echo "Directory to oracle $ORACLE_FOLDER already exists." && exit 0
# rm -Rf $ORACLE_FOLDER
if [ -f $ORACLE_FOLDER/.done ]; then

    echo "Directory to oracle: $ORACLE_FOLDER already exists --- do nothing."

else

    mkdir -p $ORACLE_FOLDER

    # copy the original AMR data: no wikification
    cp $AMR_TRAIN_FILE $ORACLE_FOLDER/ref_train.amr
    cp $AMR_DEV_FILE $ORACLE_FOLDER/ref_dev.amr
    cp $AMR_TEST_FILE $ORACLE_FOLDER/ref_test.amr

    if [[ ! "$WIKI_DEV" == "" ]]; then
        # copy the original AMR data: wiki files and original AMR with wikification
        cp $WIKI_DEV $ORACLE_FOLDER/ref_dev.wiki
        cp $WIKI_TEST $ORACLE_FOLDER/ref_test.wiki
        cp $AMR_DEV_FILE_WIKI $ORACLE_FOLDER/ref_dev.wiki.amr
        cp $AMR_TEST_FILE_WIKI $ORACLE_FOLDER/ref_test.wiki.amr
    fi

    # generate the actions

    echo -e "\nTraining data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TRAIN_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-tokens $ORACLE_FOLDER/train.en \
        --absolute-stack-positions  \
        --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
        --use-copy ${USE_COPY:-1} \
        # --reduce-nodes all

    echo -e "\nDev data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_DEV_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --out-tokens $ORACLE_FOLDER/dev.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY:-1} \
        # --reduce-nodes all

    echo -e "\nTest data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TEST_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/test.actions \
        --out-tokens $ORACLE_FOLDER/test.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY:-1} \
        # --reduce-nodes all

    touch $ORACLE_FOLDER/.done

fi
