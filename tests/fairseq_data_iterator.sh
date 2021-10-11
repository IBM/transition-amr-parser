set -o errexit 
set -o pipefail
. set_environment.sh
[ -z "$1" ] || [ -z "$2" ] && \
    echo -e "\nbash $0 <features folder> <embeddings folder>\n" && \
    exit 1
features_folder=$1
embeddings_folder=$2
set -o nounset 

python tests/fairseq_data_iterator.py \
    $features_folder  \
    --emb-dir $embeddings_folder \
    --user-dir fairseq_ext \
    --task amr_action_pointer_bartsv \
    --gen-subset train \
    --max-tokens 3584 \
    --path dummpy.pt
