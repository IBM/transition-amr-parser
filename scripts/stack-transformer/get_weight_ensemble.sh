set -o errexit
set -o pipefail
. set_environment.sh
if [ -z "$1" ];then
    model_folder=DATA/AMR/models/
else
    model_folder=$1
fi
set -o nounset


for checkpoint_best in $(find $model_folder -iname 'checkpoint_best_SMATCH.pt');do

    checkpoints_folder=$(dirname $checkpoint_best)

    [ ! -e $checkpoints_folder/checkpoint_third_best_SMATCH.pt ] && \
        echo "Have you run bash scripts/stack-transformer/rank_models.py --link-best?" && \
        exit 1

    # create average enpoint   
    python fairseq/scripts/average_checkpoints.py \
        --input \
            $checkpoints_folder/checkpoint_best_SMATCH.pt \
            $checkpoints_folder/checkpoint_second_best_SMATCH.pt \
            $checkpoints_folder/checkpoint_third_best_SMATCH.pt \
        --output $checkpoints_folder/checkpoint_top3-average_SMATCH.pt
    
done
