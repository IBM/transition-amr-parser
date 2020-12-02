set -o pipefail
set -o errexit
# setup environment
. set_environment.sh
checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\nepoch_tester.sh <checkpoints_folder>\n" && \
    exit 1
set -o nounset

[ ! -d "$checkpoints_folder" ] && \
    echo "Must be a folder $checkpoints_folder"  && \
    exit 1

# Test all existing checkpoints
for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r);do

    # pytorch model folder and basename for this checkpoints data
    model_folder=$(dirname $test_model)
    config="$model_folder/config.sh"
    std_name=$model_folder/epoch_tests/dec-$(basename $test_model .pt)

    # test checkpoint
    [ ! -f "${std_name}.actions" ] && [ -f "$test_model" ] && \
        bash scripts/stack-transformer/test.sh \
            $config \
            $test_model \
            $std_name

done

# Rank and clean-up
# After all tests are done, rank model and softlink the top 3 models according
# to the score metrics 
# model linking (will also display table)
python scripts/stack-transformer/rank_model.py \
    --link-best --no-print \
    --checkpoints $checkpoints_folder

# clean up checkpoints
python scripts/stack-transformer/remove_checkpoints.py $checkpoints_folder
