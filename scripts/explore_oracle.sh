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
    --step-by-step \
    --offset 3 \
    --clear-print
