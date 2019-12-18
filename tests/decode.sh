set -o errexit
set -o nounset
set -o pipefail 

# load local variables used below
. scripts/local_variables.sh

[ ! -d ${oracle_folder}/ ] && mkdir ${oracle_folder}/

# Use the standalone parser
amr-parse \
    --in-sentences ${oracle_folder}/dev.tokens \
    --in-model $trained_model \
    --model-config-path transition_amr_parser/config.json \
    --action-rules-from-stats ${oracle_folder}/train.rules.json \
    --num-cores 6 \
    --use-gpu \
    --batch-size 12 \
    --out-amr ${oracle_folder}/dev.amr

# evaluate model performance
echo "Evaluating Model"
test_result="$(python smatch/smatch.py --significant 3 -f $dev_file ${oracle_folder}/dev.amr -r 10)"
echo $test_result
