set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

ORACLE_FOLDER=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/oracles/o3+Word100/

# Parse a sentence step by step for debug
amr-fake-parse \
    --in-sentences ${ORACLE_FOLDER}/train.en \
    --in-actions ${ORACLE_FOLDER}/train.actions \
    --offset 3 \
    --step-by-step \
    --clear-print

# ORACLE_FOLDER=DATA/dep-parsing/oracles/PTB_SD_3_3_0+Word100/
# 
# # dependency parsing
# amr-fake-parse \
#     --in-sentences ${ORACLE_FOLDER}/train.en \
#     --in-actions ${ORACLE_FOLDER}/train.actions \
#     --step-by-step \
#     --machine-type dep-parsing \
#     --separator " " \
#     --clear-print
