set -o errexit 
set -o pipefail
. set_environment.sh
[ -z "$1" ] && \
    echo -e "\nbash $0 <features folder>\n" && \
    exit 1
features_folder=$1
set -o nounset 

python tests/stack-transformer/fairseq_data_iterator.py \
    $features_folder  \
    --gen-subset valid \
    --batch-size 128 \
    --machine-type AMR \
    --path dummpy.pt
