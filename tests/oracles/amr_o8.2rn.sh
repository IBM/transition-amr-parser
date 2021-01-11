set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
HELP="$0 <amr_file> [<target_amr>] (for evaluation)"
[ "$#" -lt 1 ] && echo "$HELP" && exit 1
train_amr=$1
if [ -z $2 ];then
    ref_smatch=""
else
    ref_smatch=$2
fi
set -o nounset

# Configuration 
MAX_WORDS=0
ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"
test_set=$(basename $train_amr)
ORACLE_TAG=o8.2_$test_set

# Variables
ORACLE_FOLDER=DATA.tests/AMR/oracles/${ORACLE_TAG}/

# Clean-up test
rm -Rf $ORACLE_FOLDER
[ ! -d $ORACLE_FOLDER/ ] && mkdir -p $ORACLE_FOLDER/

# Extract entity rules for the entire set
python scripts/extract_rules.py $train_amr $ORACLE_FOLDER/entity_rules.json

if [[ $MAX_WORDS == 0 ]]; then

    python transition_amr_parser/amr_oracle.py \
        --in-amr $train_amr \
        --in-pred-entities $ENTITIES_WITH_PREDS \
        --out-sentences $ORACLE_FOLDER/${test_set}.en \
        --out-actions $ORACLE_FOLDER/${test_set}.actions \
        --copy-lemma-action
    
else

    python transition_amr_parser/amr_oracle.py \
        --in-amr $train_amr \
        --in-pred-entities $ENTITIES_WITH_PREDS \
        --out-sentences $ORACLE_FOLDER/${test_set}.en \
        --out-actions $ORACLE_FOLDER/${test_set}.actions \
        --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
        --copy-lemma-action
    
fi

# reconstruct AMR given sentence and oracle actions without being constrained
# by training stats
python transition_amr_parser/amr_fake_parse.py \
    --in-sentences $ORACLE_FOLDER/${test_set}.en \
    --in-actions $ORACLE_FOLDER/${test_set}.actions \
    --out-amr $ORACLE_FOLDER/oracle_${test_set}.amr \
    --in-pred-entities $ENTITIES_WITH_PREDS

echo -e "\nEvaluating SMATCH (may take ~40min for AMR 2.0 train)\n"

# Compute smatch and store the results
smatch="$(smatch.py --significant 3 -r 10 -f $train_amr $ORACLE_FOLDER/oracle_${test_set}.amr)"
echo "$smatch"

# if reference provided test the score
if [ "$ref_smatch" != "" ];then
    echo "$ref_smatch"
    if [ "$smatch" != "F-score: $ref_smatch" ];then
        echo -e "[\033[91mFAILED\033[0m] Oracle ${train_amr} F-score not $ref_smatch"
        exit 1
    else
        echo -e "[\033[92mOK\033[0m] Oracle ${train_amr} test passed!"
    fi
fi
