set -o nounset
set -o pipefail 
set -o errexit 

[ ! -d scripts/ ] && echo "Call as scripts/$(basename $0)" && exit 1
. scripts/local_variables.sh

# Parse a sentence step by step for debug
amr-parse \
    --in-sentences ${oracle_folder}/train.tokens \
    --in-actions ${oracle_folder}/train.actions \
    --step-by-step \
    --offset 1552 \
    --clear-print

# - 1243 "Hong Kong Disneyland has" is confirmed after used as a node
# - 1489 "Remind me" RA(mode) without confirming me
# - 1552 "tell me" same
# - 1719 "count me" 
