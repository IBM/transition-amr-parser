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

# train_amr=$LDC2017_AMR_CORPUS/dev.txt
# ref_smatch=0.953

# Configuration 
MAX_WORDS=100
ORACLE_TAG=$(basename $train_amr)

# Variables
ORACLE_FOLDER=DATA.tests/AMR/oracles/${ORACLE_TAG}/

# Clean-up test
rm -Rf $ORACLE_FOLDER
[ ! -d $ORACLE_FOLDER/ ] && mkdir -p $ORACLE_FOLDER/

# Extract entity rules for the entire set
python scripts/extract_rules.py $train_amr $ORACLE_FOLDER/entity_rules.json

# Given AMR and sentences, create oracle actions
amr-oracle \
    --in-amr $train_amr \
    --entity-rules $ORACLE_FOLDER/entity_rules.json \
    --copy-lemma-action \
    --out-rule-stats $ORACLE_FOLDER/train.rules.json \
    --out-sentences $ORACLE_FOLDER/train.en \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-rule-stats $ORACLE_FOLDER/train.rules.json \

# Given sentences and oracle actions, recover AMR
amr-fake-parse \
    --in-sentences $ORACLE_FOLDER/train.en \
    --in-actions $ORACLE_FOLDER/train.actions \
    --entity-rules $ORACLE_FOLDER/entity_rules.json \
    --out-amr $ORACLE_FOLDER/oracle_train.amr

echo -e "\nEvaluating SMATCH (may take ~40min for AMR 2.0 train)\n"

# evaluate reconstruction performance
smatch="$(smatch.py --significant 3 -r 10 -f $train_amr $ORACLE_FOLDER/oracle_train.amr)"

echo ""
echo "$smatch"
echo "$ref_smatch"
echo ""

if [ "$ref_smatch" != "" ];then
    
    
    if [ "$smatch" != "F-score: $ref_smatch" ];then
        echo -e "[\033[91mFAILED\033[0m] Oracle ${train_amr} F-score not $ref_smatch"
        exit 1
    else
        echo -e "[\033[92mOK\033[0m] Oracle ${train_amr} test passed!"
    fi
    
fi
