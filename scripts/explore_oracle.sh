set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

ORACLE_FOLDER=DATA/AMR/oracles/o5+Word100/

# Parse a sentence step by step for debug
amr-fake-parse \
    --in-sentences ${ORACLE_FOLDER}/train.en \
    --in-actions ${ORACLE_FOLDER}/train.actions \
    --entity-rules ${ORACLE_FOLDER}/entity_rules.json \
    --offset 1 \
    --step-by-step \
    --clear-print
