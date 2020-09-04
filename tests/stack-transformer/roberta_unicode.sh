# Run small train / test on 25 sentences using same data for train/dev. Note
# that the model fails to overfit (probably due to hyperparameters or data
# size)
set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

python tests/stack-transformer/roberta.py -i DATA/roberta.tok -p roberta.large -o DATA/roberta.unicode.txt




 

