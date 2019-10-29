set -o nounset
set -o pipefail 
set -o errexit 

[ ! -d scripts/ ] && echo "Call as scripts/$(basename $0)" && exit 1
. scripts/local_variables.sh

# TRAIN
oracle_folder=data/austin0_copy_literal/
[ ! -d ${oracle_folder}/ ] && mkdir ${oracle_folder}/

# create oracle data
amr-oracle \
    --in-amr $train_file \
    --out-sentences ${oracle_folder}/train.tokens \
    --out-actions ${oracle_folder}/train.actions \
    --out-rule-stats ${oracle_folder}/train.rules.json \
    --no-whitespace-in-actions \
    # --out-amr ${oracle_folder}/train.oracle.amr \

# parse a sentence step by step
amr-parse \
    --in-sentences ${oracle_folder}/train.tokens \
    --in-actions ${oracle_folder}/train.actions \
    --out-amr ${oracle_folder}/train.amr \

# evaluate oracle performance
# wrt train.oracle.amr F-score: 0.9379
# wrt train.amr F-score: 0.9371
test_result="$(python smatch/smatch.py --significant 4 -f $train_file ${oracle_folder}/train.amr -r 10)"
if [ "$test_result" != "F-score: 0.9371" ];then
    echo $test_result
    echo "Oracle test failed! train F-score not 0.9371"
    exit 1
else:    
    echo "Train oracle test passed"
fi

# DEV

# create oracle data
echo "Generating Oracle"
amr-oracle \
    --in-amr $dev_file \
    --out-sentences ${oracle_folder}/dev.tokens \
    --out-actions ${oracle_folder}/dev.actions \
    --no-whitespace-in-actions 
    # --out-amr ${oracle_folder}/dev.oracle.amr \

# parse a sentence step by step to explore
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-actions ${oracle_folder}/dev.actions \
    --out-amr ${oracle_folder}/dev.amr \
    # --action-rules-from-stats ${oracle_folder}/train.rules.json \

# evaluate oracle performance
echo "Evaluating Oracle"
# wrt dev.oracle.amr F-score: 0.9381
# wrt dev.amr F-score: 0.9379
test_result="$(python smatch/smatch.py --significant 4 -f $dev_file ${oracle_folder}/dev.amr -r 10)"
if [ "$test_result" != "F-score: 0.9379" ];then
    echo $test_result
    echo "Oracle test failed! train F-score not 0.9378"
    exit 1
else:    
    echo "Dev oracle test passed"
fi
