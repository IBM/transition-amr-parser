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
    --no-whitespace-in-actions \
    --out-rule-stats ${oracle_folder}/train.rules.json \
    #--out-amr ${oracle_folder}/train.oracle.amr \

# parse a sentence step by step
amr-parse \
    --in-sentences ${oracle_folder}/train.tokens \
    --in-actions ${oracle_folder}/train.actions \
    --out-amr ${oracle_folder}/train.amr \

# evaluate oracle performance
# wrt train.oracle.amr F-score: 0.9379
# wrt train.amr F-score: 0.9371
test_result="$(python smatch/smatch.py --significant 4 -f $train_file ${oracle_folder}/train.amr -r 10)"
echo $test_result
if [ "$test_result" != "F-score: 0.9371" ];then
    echo "Oracle train test failed! train F-score not 0.9371"
    exit 1
else
    echo "Oracle train test passed!"
fi

# DEV

# ATTENTION: To pass the tests the dev test must have alignments as those
# obatined with the preprocessing described in README

# create oracle data
echo "Generating Oracle"
amr-oracle \
    --in-amr $dev_file \
    --out-sentences ${oracle_folder}/dev.tokens \
    --out-actions ${oracle_folder}/dev.actions \
    --no-whitespace-in-actions \
#    --out-amr ${oracle_folder}/dev.oracle.amr \

# parse a sentence step by step to explore
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-actions ${oracle_folder}/dev.actions \
    --out-amr ${oracle_folder}/dev.amr \

# evaluate oracle performance
echo "Evaluating Oracle"
test_result="$(python smatch/smatch.py --significant 3 -f $dev_file ${oracle_folder}/dev.amr -r 10)"
echo $test_result
if [ "$test_result" != "F-score: 0.938" ];then
    echo "Oracle dev test failed! train F-score not 0.938"
    exit 1
else
    echo "Oracle dev test passed!"
fi

# parse a sentence step by step to explore
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-actions ${oracle_folder}/dev.actions \
    --out-amr ${oracle_folder}/dev.amr \
    --action-rules-from-stats ${oracle_folder}/train.rules.json \

# evaluate oracle performance
echo "Evaluating Oracle"
test_result="$(python smatch/smatch.py --significant 3 -f $dev_file ${oracle_folder}/dev.amr -r 10)"
echo $test_result
if [ "$test_result" != "F-score: 0.914" ];then
    echo "Oracle dev test failed! train F-score not 0.914"
    exit 1
else
    echo "Oracle dev test passed!"
fi
