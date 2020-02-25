set -o pipefail
set -o errexit
# setup environment
. set_environment.sh
checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\nepoch_tester.sh <model_checkpoint>\n" && \
    exit 1
set -o nounset

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
    echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}"
    fairseq-generate $FAIRSEQ_GENERATE_ARGS --quiet --path $test_model --results-path ${std_name}
    
    # Create the AMR from the model obtained actions
    amr-fake-parse \
        --in-sentences $ORACLE_FOLDER/dev.en \
        --in-actions ${std_name}.actions \
        --out-amr ${std_name}.amr 

    # add wiki
    python fairseq/dcc/add_wiki.py \
        ${std_name}.amr $WIKI_DEV \
        > ${std_name}.wiki.amr
        
    # compute smatch wrt gold with wiki
    python smatch/smatch.py \
         --significant 4  \
         -f $AMR_DEV_FILE_WIKI \
         ${std_name}.wiki.amr \
         -r 10 \
         > ${std_name}.wiki.smatch
    
    # plot score
    cat ${std_name}.wiki.smatch
    
done
