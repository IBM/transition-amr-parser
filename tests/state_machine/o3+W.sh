set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
[ -z "$1" ] && echo "$0 train (or dev)" && exit 1
test_set=$1
set -o nounset

# Configuration 
MAX_WORDS=100
ORACLE_TAG=o3+Word${MAX_WORDS}
ORACLE_FOLDER=DATA.tests/AMR/oracles/${ORACLE_TAG}/

# Select between train/dev
train_amr=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
if [ "$test_set" == "dev" ];then
    # ATTENTION: To pass the tests the dev test must have alignments as those
    # obtained with the preprocessing described in README
    reference_amr=$LDC2017_AMR_CORPUS/dev.txt
    # Do not limit actions by rules
    ref_smatch=0.938
    # limit actions by rules
    ref_smatch2=0.921
elif [ "$test_set" == "train" ];then
    reference_amr=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt
    ref_smatch=0.937
else
    echo "Usupported set $test"
    exit 1
fi

# TRAIN
[ ! -d $ORACLE_FOLDER/ ] && mkdir -p $ORACLE_FOLDER/


# create oracle actions from AMR and the sentence for the train set. This also
# accumulates necessary statistics in train.rules.json
if [ ! -f "$ORACLE_FOLDER/train.rules.json" ];then
    amr-oracle \
        --in-amr $train_amr \
        --out-sentences $ORACLE_FOLDER/train.en \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-rule-stats $ORACLE_FOLDER/train.rules.json \
        --multitask-max-words $MAX_WORDS  \
        --out-multitask-words $ORACLE_FOLDER/train.multitask_words \
        --copy-lemma-action  
fi

# create oracle actions from AMR and the sentence for the dev set if needed
if [ ! -f "$ORACLE_FOLDER/${test_set}.rules.json" ];then
    amr-oracle \
        --in-amr $reference_amr \
        --out-sentences $ORACLE_FOLDER/${test_set}.en \
        --out-actions $ORACLE_FOLDER/${test_set}.actions \
        --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
        --copy-lemma-action
fi

# reconstruct AMR given sentence and oracle actions without being constrained
# by training stats
amr-fake-parse \
    --in-sentences $ORACLE_FOLDER/${test_set}.en \
    --in-actions $ORACLE_FOLDER/${test_set}.actions \
    --out-amr $ORACLE_FOLDER/oracle_${test_set}.amr

# evaluate reconstruction performance
smatch="$(smatch.py --significant 3 -r 10 -f $reference_amr $ORACLE_FOLDER/oracle_${test_set}.amr)"

echo "$smatch"
echo "$ref_smatch"

if [ "$smatch" != "F-score: $ref_smatch" ];then
    echo -e "[\033[91mFAILED\033[0m] Oracle ${test_set} F-score not $ref_smatch"
    exit 1
else
    echo -e "[\033[92mOK\033[0m] Oracle ${test_set} test passed!"
fi

# reconstruct AMR given sentence and oracle actions using train statistics to
# define rules (real case)
if [ "$test_set" == "dev" ];then

    amr-fake-parse \
        --in-sentences $ORACLE_FOLDER/${test_set}.en \
        --in-actions $ORACLE_FOLDER/${test_set}.actions \
        --action-rules-from-stats $ORACLE_FOLDER/train.rules.json \
        --out-amr $ORACLE_FOLDER/oracle_${test_set}.rules.amr
    
    # evaluate reconstruction performance
    echo "Evaluating Oracle"
    smatch="$(smatch.py --significant 3 -r 10 -f $reference_amr $ORACLE_FOLDER/oracle_${test_set}.rules.amr)"

    echo "$smatch"
    echo "$ref_smatch2"
    
    # chek if test passed
    if [ "$smatch" != "F-score: $ref_smatch2" ];then
        echo -e "[\033[91mFAILED\033[0m] Oracle ${test_set} F-score not $ref_smatch2"
        exit 1
    else
        echo -e "[\033[92mOK\033[0m] Oracle ${test_set} test passed!"
    fi
fi
