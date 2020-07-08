# Create files neccesary for standalone in v0.3.0. This happens automaticaly in
# training. This script is for old models.
set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

for checkpoint_config in ${@:1};do

    # Sanity checks
    [ "$(basename $checkpoint_config)" != "config.sh" ] && \
        echo "Expected config under $checkpoint_config" && \
        exit 1

    # load config
    . $checkpoint_config

    # get model folder
    checkpoints_dir=$(dirname $checkpoint_config)

    # Copy dictinoaries
    cp $features_folder/dict.*.txt $checkpoints_dir/

    # store the preprocessing and training parameters. We will need this to
    # know which roberta config we used
    python scripts/stack-transformer/save_fairseq_args.py \
        --fairseq-preprocess-args "$FAIRSEQ_PREPROCESS_ARGS" \
        --fairseq-train-args "$FAIRSEQ_TRAIN_ARGS" \
        --out-fairseq-model-config $checkpoints_dir/config.json

done
