set -o errexit 
set -o pipefail
. set_environment.sh
[ -z "$1" ] && \
    echo -e "\nbash $0 <features folder>\n" && \
    exit 1
FEATURES_FOLDER=$1
set -o nounset 

python tests/stack-transformer/fairseq_data_iterator.py \
    $FEATURES_FOLDER  \
    --gen-subset train \
    --max-tokens 3584 \
    --machine-type AMR \
    --path dummpy.pt
