set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

for checkpoints_folder in "$@";do

    # this should be a folder containing checkpoints
    [ ! -f "$checkpoints_folder/checkpoint_last.pt" ] && \
        echo "Expected $checkpoints_folder/checkpoint_last.pt" && \
        exit 1

    # CREATE ENSEMBLE
    ensemble_checkpoint=$checkpoints_folder/checkpoint_top3-average.pt
    if [ ! -f "$ensemble_checkpoint" ];then

        # softlink top 3 models
        [ ! -e $checkpoints_folder/checkpoint_third_best_SMATCH.pt ] && \
            python scripts/stack-transformer/rank_model.py --link-best --no-print

        # create average enpoint   
        python fairseq/scripts/average_checkpoints.py \
            --input \
                $checkpoints_folder/checkpoint_best_SMATCH.pt \
                $checkpoints_folder/checkpoint_second_best_SMATCH.pt \
                $checkpoints_folder/checkpoint_third_best_SMATCH.pt \
            --output $ensemble_checkpoint
    fi

    # TEST ENSEMBLE
    # create config copy labeled as using weight ensemble
    cp $checkpoints_folder/config.sh $checkpoints_folder/config_top3-average.sh
    sed 's@^TEST_TAG=.*@TEST_TAG="top3-average"@' -i $checkpoints_folder/config_top3-average.sh
    echo "Created $checkpoints_folder/config_top3-average.sh"

    # run test
    [ -f "$checkpoints_folder/top3-average/valid.actions" ] && continue
    bash scripts/stack-transformer/test.sh $checkpoints_folder/config_top3-average.sh $ensemble_checkpoint
    
done
