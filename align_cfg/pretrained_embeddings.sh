set -o errexit
set -o pipefail
. set_environment.sh
# this requires a special environment with allennlp
conda deactivate
[ ! -d cenv_ELMO ] && echo -e "\nbash align_cfg/install.sh ?\n" && exit 1
conda activate ./cenv_ELMO

export PYTHONPATH=.

[ -z "$1" ] && echo -e "\n$0 /path/to/embeddings/ (where ELMO_vocab.<amr|text>.txt are) \n" 
FOLDER=$1

set -o nounset

python align_cfg/pretrained_embeddings.py --cuda \
    --vocab-text $FOLDER/ELMO_vocab.text.txt
python align_cfg/pretrained_embeddings.py --cuda \
    --vocab-text $FOLDER/ELMO_vocab.amr.txt
