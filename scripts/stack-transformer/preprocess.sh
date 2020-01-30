set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

# Argument handling
config=$1

# Load config
. "$config"

# stage-1: Preprocess

# ORACLE
[ ! -d $ORACLE_FOLDER ] && mkdir -p $ORACLE_FOLDER

# Create oracle data
# Train
amr-oracle \
    --in-amr $AMR_TRAIN_FILE \
    --out-sentences $ORACLE_FOLDER/train.en \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-rule-stats $ORACLE_FOLDER/train.rules.json \
    $ORACLE_TRAIN_ARGS

# Dev and test
amr-oracle \
    --in-amr $AMR_DEV_FILE \
    --out-sentences $ORACLE_FOLDER/dev.en \
    --out-actions $ORACLE_FOLDER/dev.actions \
    --out-rule-stats $ORACLE_FOLDER/dev.rules.json \
    $ORACLE_DEV_ARGS

amr-oracle \
    --in-amr $AMR_TEST_FILE \
    --out-sentences $ORACLE_FOLDER/test.en \
    --out-actions $ORACLE_FOLDER/test.actions \
    --out-rule-stats $ORACLE_FOLDER/test.rules.json \
    $ORACLE_DEV_ARGS

# PREPROCESSING
# extract data
echo "fairseq-preprocess $FAIRSEQ_PREPROCESS_ARGS"
fairseq-preprocess $FAIRSEQ_PREPROCESS_ARGS 
