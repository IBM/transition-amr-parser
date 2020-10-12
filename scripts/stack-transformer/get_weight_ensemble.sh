set -o errexit
set -o pipefail
# Argument handling
HELP="$0 <score name> <model folder 1> [<model folder 2> ...]"
[ -z "$1" ] && echo $HELP  && exit 1
score_name=$1
[ "$#" -lt 2 ] && echo $HELP  && exit 1
# load virtual envs
. set_environment.sh
set -o nounset

for checkpoints_folder in "${@:2}";do

    # this should be a folder containing checkpoints
    [ ! -f "$checkpoints_folder/checkpoint_last.pt" ] && \
        echo "Expected $checkpoints_folder/checkpoint_last.pt" && \
        exit 1

    # force the user to call the ranker themselves
    # i.e. python scripts/stack-transformer/rank_model.py --link-best --no-print 
    [ ! -e $checkpoints_folder/checkpoint_third_best_${score_name}.pt ] && \
        echo "Expected $checkpoints_folder/checkpoint_third_best_${score_name}.pt" && \
        exit 1

    # CREATE ENSEMBLE
    ensemble_checkpoint=$checkpoints_folder/checkpoint_top3-average.pt
    if [ ! -f "$ensemble_checkpoint" ];then

        # create average enpoint   
        python fairseq/scripts/average_checkpoints.py \
            --input \
                $checkpoints_folder/checkpoint_best_${score_name}.pt \
                $checkpoints_folder/checkpoint_second_best_${score_name}.pt \
                $checkpoints_folder/checkpoint_third_best_${score_name}.pt \
            --output $ensemble_checkpoint
    fi

    # TEST ENSEMBLE
    [ -f "$checkpoints_folder/top3-average/valid.actions" ] && continue
    # create config copy labeled as using weight ensemble
    cp $checkpoints_folder/config.sh $checkpoints_folder/config_top3-average.sh
    sed 's@^TEST_TAG=.*@TEST_TAG="top3-average"@' -i $checkpoints_folder/config_top3-average.sh
    echo "Created $checkpoints_folder/config_top3-average.sh"

    # run test
    bash scripts/stack-transformer/test.sh $checkpoints_folder/config_top3-average.sh $ensemble_checkpoint
    
done
