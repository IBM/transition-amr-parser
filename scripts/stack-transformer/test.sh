set -o errexit 
set -o pipefail
# set -o nounset 

# Argument handling
config=$1
test_model=$2

[ -z "$config" ] && echo -e "\ntest.sh <config> <model_checkpoint>\n" && exit 1
[ -z "$test_model" ] && echo -e "\ntest.sh <config> <model_checkpoint>\n" && exit 1

# Load config
. "$config"

# setup environment
. set_environment.sh

# this is given by calling script to iterate over seeds

# decode 
echo "fairseq-generate $fairseq_generate_args --path $test_model"
fairseq-generate $fairseq_generate_args --path $test_model
# to profile decoder
# pip install line_profiler
# decorate target function with @profile
# call instead of fairseq-generate
# kernprof -l generate.py $fairseq_generate_args --path $test_model
# then you can consult details with 
# python -m line_profiler generate.py.lprof

model_folder=$(dirname $test_model)

# FIXME: We linked dev to test to handle fairseqs hardwired variables. Probably
# will come to bite us in the future
amr-parse \
    --in-sentences $extracted_oracle_folder/${data_set}_extracted/dev.en \
    --in-actions $model_folder/test.actions \
    --out-amr $model_folder/test.amr \

python smatch/smatch.py \
     --significant 4  \
     -f $amr_dev_file \
     $model_folder/test.amr \
     -r 10 \
     > $model_folder/test.smatch

cat $model_folder/test.smatch
