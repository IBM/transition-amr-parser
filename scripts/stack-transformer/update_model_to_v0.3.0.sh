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
        echo "$checkpoint_config should be a path to a config.sh" && \
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

    # entity_rules.json is not more in code and needs to be created during
    # training. Copy the old entity_rules.json available in the code to the
    # model forlder to update the model to v0.3.0+
    [ ! -f "$checkpoints_dir/entity_rules.json" ] && \
        echo "\033[93mPlease manually add a entity_rules.json to $checkpoints_dir\033[0m" && \
        exit 1

done
