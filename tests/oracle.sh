set -o nounset
set -o pipefail 
set -o errexit 

[ ! -d scripts/ ] && echo "Call as scripts/$(basename $0)" && exit 1
. scripts/local_variables.sh

# TRAIN
[ ! -d ${oracle_folder}/ ] && mkdir ${oracle_folder}/

# create oracle data
amr-oracle \
    --in-amr $train_file \
    --out-sentences ${oracle_folder}/train.tokens \
    --out-actions ${oracle_folder}/train.actions \
    --out-rule-stats ${oracle_folder}/train.rules.json \
    #--no-whitespace-in-actions \
    #--out-amr ${oracle_folder}/train.oracle.amr \

# parse a sentence step by step
amr-parse \
    --in-sentences ${oracle_folder}/train.tokens \
    --in-actions ${oracle_folder}/train.actions \
    --out-amr ${oracle_folder}/train.amr \
    --action-rules-from-stats ${oracle_folder}/train.rules.json  \

# evaluate oracle performance
# wrt train.oracle.amr F-score: 0.9379
# wrt train.amr F-score: 0.9371
# F-score: valid actions 0.9366
# F-score: valid actions + possible predicted 0.9365
test_result="$(python smatch/smatch.py --significant 4 -f $train_file ${oracle_folder}/train.amr -r 10)"
echo $test_result
ref_score=0.9366
if [ "$test_result" != "F-score: $ref_score" ];then
    printf "[\033[91mFAILED\033[0m] Oracle train F-score not $ref_score\n"
    exit 1
else
    printf "[\033[92mOK\033[0m] Oracle train passed!\n"
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
    --out-rule-stats ${oracle_folder}/dev.rules.json 

# parse a sentence step by step to explore
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-actions ${oracle_folder}/dev.actions \
    --out-amr ${oracle_folder}/dev.amr \

# evaluate oracle performance
echo "Evaluating Oracle"
test_result="$(python smatch/smatch.py --significant 3 -f $dev_file ${oracle_folder}/dev.amr -r 10)"
echo $test_result
ref_score=0.938
if [ "$test_result" != "F-score: $ref_score" ];then
    echo -e "[\033[91mFAILED[0m] Oracle dev test F-score not $ref_score"
    exit 1
else
    echo -e "[\033[92mOK\033[0m] Oracle dev test passed!"
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
ref_score=0.915
if [ "$test_result" != "F-score: $ref_score" ];then
    printf "[\033[91mFAILED\033[0m] Oracle dev test F-score not $ref_score\n"
    exit 1
else
    printf "[\033[92mOK\033[0m] Oracle dev test passed!\n"
fi
