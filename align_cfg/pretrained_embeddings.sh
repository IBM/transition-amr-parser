set -o errexit
set -o pipefail
. set_environment.sh
# this requires a special environment with allennlp
[ -z "$1" ] && echo -e "\n$0 /path/to/embeddings/ (where vocab.<amr|text>.txt are) \n"
FOLDER=$1

set -o nounset

python align_cfg/pretrained_embeddings.py --cuda --allow-cpu \
    --vocab $FOLDER/vocab.text.txt \
    --cache-dir $FOLDER/
python align_cfg/pretrained_embeddings.py --cuda --allow-cpu \
    --vocab $FOLDER/vocab.amr.txt \
    --cache-dir $FOLDER/
