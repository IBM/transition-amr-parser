set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
set -o nounset

# TRAIN
[ ! -d $ORACLE_FOLDER/ ] && mkdir -p $ORACLE_FOLDER/

# DEV
MAX_WORDS=100
ORACLE_TAG=o3+Word${MAX_WORDS}
ORACLE_FOLDER=DATA/AMR/oracles/${ORACLE_TAG}/

# ATTENTION: To pass the tests the dev test must have alignments as those
# obtained with the preprocessing described in README
reference_amr=$LDC2017_AMR_CORPUS/dev.txt

# create oracle data
echo "Generating Oracle Actions"
amr-oracle \
    --in-amr $reference_amr \
    --out-sentences $ORACLE_FOLDER/dev.en \
    --out-actions $ORACLE_FOLDER/dev.actions \
    --multitask-max-words $MAX_WORDS  \
    --out-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action 
    # --out-rule-stats $ORACLE_FOLDER/dev.rules.json 

# parse a sentence step by step to explore
amr-fake-parse \
    --in-sentences $ORACLE_FOLDER/dev.tokens \
    --in-actions $ORACLE_FOLDER/dev.actions \
    --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action \
    --out-amr $ORACLE_FOLDER/dev.amr

# evaluate oracle performance
echo "Evaluating Oracle"
dev_result="$(smatch.py --significant 3 -f $reference_amr $ORACLE_FOLDER/dev.amr -r 10)"
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
    --in-sentences $ORACLE_FOLDER/dev.tokens \
    --in-actions $ORACLE_FOLDER/dev.actions \
    --out-amr $ORACLE_FOLDER/dev.amr \
    --action-rules-from-stats $ORACLE_FOLDER/train.rules.json \

# evaluate oracle performance
echo "Evaluating Oracle"
dev_result="$(python smatch/smatch.py --significant 3 -f $reference_amr $ORACLE_FOLDER/dev.amr -r 10)"
echo $dev_result
ref_score=0.915
if [ "$dev_result" != "F-score: $ref_score" ];then
    printf "[\033[91mFAILED\033[0m] Oracle dev test F-score not $ref_score\n"
    exit 1
else
    printf "[\033[92mOK\033[0m] Oracle dev test passed!\n"
fi
