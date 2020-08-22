#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
set -o nounset


##### CONFIG
dir=$(dirname $0)
if [ ! -z "$1" ]; then
    config=$1
    . $dir/$config    # we should always call from one level up
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

##### reconstruct AMR given sentence and oracle actions without being constrained
# by training stats
for split in dev test train
do

    if [ $split == "dev" ]; then
        ref=$AMR_DEV_FILE
    elif [ $split == "test" ]; then
        ref=$AMR_TEST_FILE
    else
        ref=$AMR_TRAIN_FILE
    fi
    
    if [ ! -f $ORACLE_FOLDER/oracle_$split.smatch ]; then

        python transition_amr_parser/o7_fake_parse.py \
            --in-sentences $ORACLE_FOLDER/$split.en \
            --in-actions $ORACLE_FOLDER/$split.actions \
            --out-amr $ORACLE_FOLDER/oracle_$split.amr

        ##### evaluate reconstruction performance
        # smatch="$(smatch.py --significant 3 -r 10 -f $reference_amr $ORACLE_FOLDER/oracle_${test_set}.amr)"

        smatch.py --significant 3 -r 10 -f $ref $ORACLE_FOLDER/oracle_$split.amr > $ORACLE_FOLDER/oracle_$split.smatch
    
    else
    
        echo "smatch result for "$ORACLE_FOLDER/oracle_$split.amr" already exists."
    
    fi

    cat $ORACLE_FOLDER/oracle_$split.smatch

done

