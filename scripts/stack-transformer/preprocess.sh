set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

# load config
config=$1
. "$config"

# stage-1: Preprocess

# ORACLE
[ ! -d $oracle_folder ] && mkdir -p $oracle_folder

# Labeled shift: each time we shift, we also predict the word being shited
# use top MAX_WORDS 
# --multitask-max-words --out-multitask-words --in-multitask-words
MAX_WORDS=100

# To have an action calling external lemmatizer
# --copy-lemma-action

# Create oracle data
amr-oracle \
    --in-amr $amr_train_file \
    --out-sentences $oracle_folder/train.en \
    --out-actions $oracle_folder/train.actions \
    --multitask-max-words $MAX_WORDS \
    --out-multitask-words $oracle_folder/train.multitask_words \
    --out-rule-stats $oracle_folder/train.rules.json \
    --copy-lemma-action

amr-oracle \
    --in-amr $amr_dev_file \
    --in-multitask-words $oracle_folder/train.multitask_words \
    --out-sentences $oracle_folder/dev.en \
    --out-actions $oracle_folder/dev.actions \
    --out-rule-stats $oracle_folder/dev.rules.json \
    --copy-lemma-action

amr-oracle \
    --in-amr $amr_test_file \
    --in-multitask-words $oracle_folder/train.multitask_words \
    --out-sentences $oracle_folder/test.en \
    --out-actions $oracle_folder/test.actions \
    --out-rule-stats $oracle_folder/test.rules.json \
    --copy-lemma-action

# PREPROCESSING
# extract data
echo "fairseq-preprocess $fairseq_preprocess_args"
fairseq-preprocess $fairseq_preprocess_args 
