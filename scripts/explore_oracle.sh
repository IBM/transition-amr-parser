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
    --out-amr ${oracle_folder}/train.oracle.amr \
    --out-sentences ${oracle_folder}/train.tokens \
    --out-actions ${oracle_folder}/train.actions \
    --out-rule-stats ${oracle_folder}/train.rules.json \
    --no-whitespace-in-actions \

# parse a sentence step by step
amr-parse \
    --in-sentences ${oracle_folder}/train.tokens \
    --in-actions ${oracle_folder}/train.actions \
    --out-amr ${oracle_folder}/train.amr \

# parse a sentence step by step
# amr-parse \
#     --in-sentences ${oracle_folder}/train.tokens \
#     --in-actions ${oracle_folder}/train.actions \
#     --out-amr tmp \
#     --step-by-step \
#     --offset 34435 \
#     --clear-print

# evaluate oracle performance
# old: F-score: 0.9379
# new: F-score: 0.9371
python smatch/smatch.py \
     --significant 4  \
     -f $train_file \
     ${oracle_folder}/train.amr \
     -r 10 

# DEV

# create oracle data
echo "Generating Oracle"
amr-oracle \
    --in-amr $dev_file \
    --out-amr ${oracle_folder}/dev.oracle.amr \
    --out-sentences ${oracle_folder}/dev.tokens \
    --out-actions ${oracle_folder}/dev.actions \
    --no-whitespace-in-actions 

# parse a sentence step by step to explore
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-actions ${oracle_folder}/dev.actions \
    --out-amr ${oracle_folder}/dev.amr \
    # --action-rules-from-stats ${oracle_folder}/train.rules.json \

# evaluate oracle performance
echo "Evaluating Oracle"
# old: F-score: 0.9381
# new: F-score: 0.9378
python smatch/smatch.py \
     --significant 4  \
     -f $dev_file \
     ${oracle_folder}/dev.amr \
     -r 10 
