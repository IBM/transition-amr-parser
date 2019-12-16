#
# TODO: Standalone call of parser
#
set -o errexit
set -o nounset
set -o pipefail 

config=$1

# load local variables used below
. $config 

[ ! -d ${output_folder}/ ] && mkdir ${output_folder}/

# Use the stanalone parser
amr-parse \
    --in-sentences ${output_folder}/dev.tokens \
    --in-model $trained_model \
    --model-config-path $config_path \
    --action-rules-from-stats ${oracle_stats} \
    --num-cores 6 \
    --use-gpu \
    --batch-size 12 \
    --out-amr ${output_folder}/dev.amr

# evaluate model performance
echo "Evaluating Model"
test_result="$(python smatch/smatch.py --significant 3 -f $dev_file ${output_folder}/dev.amr -r 10)"
echo $test_result
