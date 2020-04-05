set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

ORACLE_FOLDER=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/oracles/o3+Word100/

# parse a sentence step by step
amr-fake-parse \
    --in-sentences $ORACLE_FOLDER/dev.en \
    --in-actions $ORACLE_FOLDER/dev.actions \
    --out-amr tmp.amr \
    --action-rules-from-stats $ORACLE_FOLDER/dev.rules.json

amr-edit --in-amr tmp.amr --out-amr tmp2.amr 

vimdiff tmp.amr tmp2.amr
