# Run small train / test on 25 sentences using same data for train/dev. Note
# that the model fails to overfit (probably due to hyperparameters or data
# size)
set -o errexit
set -o pipefail
# default AMR is wiki 25 sentences
if [ -z "$1" ];then
    amr_file=DATA/wiki25.jkaln
else    
    amr_file=$1
fi    
. set_environment.sh
set -o nounset

mkdir -p DATA.tests/roberta/

# create tokens from wiki task
echo "Creating DATA.tests/roberta/roberta.tok from $amr_file"
grep '::tok' $amr_file | sed 's@.*::tok @@' > DATA.tests/roberta/roberta.tok

# Use this one if you do not have tokenized sentences (unicode should fail equally)
# grep '::snt' $amr_file | sed 's@.*::snt@@' > DATA.tests/roberta/roberta.tok

echo "Extracting DATA.tests/roberta/roberta.tok"
python tests/stack-transformer/roberta.py \
    -i DATA.tests/roberta/roberta.tok \
    -p roberta.large \
    -o DATA.tests/roberta/roberta.unicode.txt \
    --raise-error

# If we reach here we passed
echo -e "[\033[92mOK\033[0m] RoBERTA extraction for $amr_file works"
