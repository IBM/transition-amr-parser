set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
set -o nounset

# TRAIN
[ ! -d DATA/oracles/basic/ ] && mkdir -p DATA/oracles/basic/

# create oracle data
train_file=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
amr-oracle --in-amr $train_file \
    --out-sentences DATA/oracles/basic/train.tokens \
    --out-actions DATA/oracles/basic/train.actions \
    --out-rule-stats DATA/oracles/basic/train.rules.json

# parse a sentence step by step
amr-fake-parse \
    --in-sentences DATA/oracles/basic/train.tokens \
    --in-actions DATA/oracles/basic/train.actions \
    --out-amr DATA/oracles/basic/train.amr \
    --action-rules-from-stats DATA/oracles/basic/train.rules.json

# evaluate oracle performance
# wrt train.oracle.amr F-score: 0.9379
# wrt train.amr F-score: 0.9371
# F-score: valid actions 0.9366
# F-score: valid actions + possible predicted 0.9365
test_result="$(python smatch/smatch.py --significant 4 -f $train_file DATA/oracles/basic/train.amr -r 10)"
echo $test_result
ref_score=0.9365
if [ "$test_result" != "F-score: $ref_score" ];then
    printf "[\033[91mFAILED\033[0m] Oracle train F-score not $ref_score\n"
    exit 1
else
    printf "[\033[92mOK\033[0m] Oracle train passed!\n"
fi

# DEV

# ATTENTION: To pass the tests the dev test must have alignments as those
# obtained with the preprocessing described in README

# create oracle data
echo "Generating Oracle"
dev_file=$LDC2017_AMR_CORPUS/dev.txt
amr-oracle \
    --in-amr $dev_file \
    --out-sentences DATA/oracles/basic/dev.tokens \
    --out-actions DATA/oracles/basic/dev.actions \
    --out-rule-stats DATA/oracles/basic/dev.rules.json 

# parse a sentence step by step to explore
amr-fake-parse \
    --in-sentences DATA/oracles/basic/dev.tokens \
    --in-actions DATA/oracles/basic/dev.actions \
    --out-amr DATA/oracles/basic/dev.amr \

# evaluate oracle performance
echo "Evaluating Oracle"
dev_result="$(python smatch/smatch.py --significant 3 -f $dev_file DATA/oracles/basic/dev.amr -r 10)"
echo $dev_result
ref_score=0.938
if [ "$dev_result" != "F-score: $ref_score" ];then
    echo -e "[\033[91mFAILED[0m] Oracle dev test F-score not $ref_score"
    exit 1
else
    echo -e "[\033[92mOK\033[0m] Oracle dev test passed!"
fi

# parse a sentence step by step to explore
amr-fake-parse \
    --in-sentences DATA/oracles/basic/dev.tokens \
    --in-actions DATA/oracles/basic/dev.actions \
    --out-amr DATA/oracles/basic/dev.amr \
    --action-rules-from-stats DATA/oracles/basic/train.rules.json \

# evaluate oracle performance
echo "Evaluating Oracle"
dev_result="$(python smatch/smatch.py --significant 3 -f $dev_file DATA/oracles/basic/dev.amr -r 10)"
echo $dev_result
ref_score=0.915
if [ "$dev_result" != "F-score: $ref_score" ];then
    printf "[\033[91mFAILED\033[0m] Oracle dev test F-score not $ref_score\n"
    exit 1
else
    printf "[\033[92mOK\033[0m] Oracle dev test passed!\n"
fi
