#set -o nounset
#set -o pipefail
#set -o errexit
#set -o xtrace

# seconds
SLEEP_TIME=60

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\nepoch_tester.sh <model_checkpoint>\n" && \
    exit 1

# setup environment
. set_environment.sh

# Loop over existing checkpoints
for test_model in $(find $checkpoints_folder -iname 'checkpoint[0-9]*.pt' | sort -r);do

    # pytorch model folder and basename for this checkpoints data
    model_folder=$(dirname $test_model)
    output_folder="$model_folder/epoch_tests/"
    config="$model_folder/config.sh"
    std_name=$output_folder/dec-$(basename $test_model .pt)

    # logs for each run of the checkpoint will be stored here
    mkdir -p "$output_folder"

    # skip if log exists
    if [ -f "${std_name}.smatch" ];then
        echo -e "Skipping $std_name"
        cat ${std_name}.smatch
        continue
    fi

    # activate config
    . "$config"

    # decode 
    if [ ! -f "$test_model" ];then
        # model was meanwhile deleted
        continue
    fi
    echo "fairseq-generate $fairseq_generate_args --quiet --path $test_model --results-path ${std_name}"
    fairseq-generate $fairseq_generate_args --quiet --path $test_model --results-path ${std_name}
    
    # will come to bite us in the future
    amr-fake-parse \
        --in-sentences $extracted_oracle_folder/${data_set}_extracted/dev.en \
        --in-actions ${std_name}.actions \
        --out-amr ${std_name}.amr 
        
    python smatch/smatch.py \
         --significant 4  \
         -f $amr_dev_file \
         ${std_name}.amr \
         -r 10 \
         > ${std_name}.smatch
    
    # plot score
    cat ${std_name}.smatch
    
done
