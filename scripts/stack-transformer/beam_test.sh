set -o pipefail
set -o errexit
# setup environment
. set_environment.sh
checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\n$0 <checkpoints_folder>\n" && \
    exit 1
set -o nounset

[ ! -d "$checkpoints_folder" ] && \
    echo "Must be a folder $checkpoints_folder"  && \
    exit 1

config="$checkpoints_folder/config.sh"
[ ! -f $config ] && echo "Missing $config" && exit 1 

# load config
. $config

# Right now do ensemble and beam automatically only for AMR
if [ "$TASK_TAG" == "AMR" ];then

    if [ "$WIKI_DEV" == "" ];then
        score_name="SMATCH"
    else
        score_name="WIKI.SMATCH"
    fi

    # checkpoint averaging of best three checkpoints
    [ ! -f "$checkpoints_folder/top3-average/valid.actions" ] && \
        bash scripts/stack-transformer/get_weight_ensemble.sh \
            $score_name \
            $checkpoints_folder

    # beam 10 tests
    beam_size=10
    [ ! -f "$checkpoints_folder/top3-average_beam${beam_size}/valid.actions" ] && \
        bash scripts/stack-transformer/get_beam_test.sh \
            ${beam_size} \
            top3-average \
            $checkpoints_folder/checkpoint_top3-average.pt

fi
