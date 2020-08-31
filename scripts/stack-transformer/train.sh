set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset 

# Argument handling
config=$1
seed=$2

# Load config
. "$config"

dir=$(dirname $0)
parentdir="$(dirname "$dir")"

# this is given by calling script to iterate over seeds
checkpoints_dir="${CHECKPOINTS_DIR_ROOT}-seed${seed}/"

[ ! -d "$checkpoints_dir" ] && \
    mkdir -p "$checkpoints_dir"

# If rules were used, copy them to model folder
if [ -f "$ORACLE_FOLDER/train.rules.json" ];then
    echo "cp $ORACLE_FOLDER/train.rules.json $checkpoints_dir"
    cp $ORACLE_FOLDER/train.rules.json $checkpoints_dir
fi

# store the preprocessing and training parameters. We will need this to
# know which roberta config we used
python scripts/stack-transformer/save_fairseq_args.py \
    --fairseq-preprocess-args "$FAIRSEQ_PREPROCESS_ARGS" \
    --fairseq-train-args "$FAIRSEQ_TRAIN_ARGS" \
    --out-fairseq-model-config $checkpoints_dir/config.json

if [ "$TASK_TAG" == "AMR" ];then
    # Copy entity_rules.json from oracle, created using train file
    if [ -n "${ENTITY_RULES:-}" ] && [ -f "$ENTITY_RULES" ]; then
        cp $ENTITY_RULES $checkpoints_dir
    else
        if [ -f "$ORACLE_FOLDER/entity_rules.json" ];then
    	    cp $ORACLE_FOLDER/entity_rules.json $checkpoints_dir
        else
    	    python $parentdir/extract_rules.py $AMR_TRAIN_FILE $checkpoints_dir/entity_rules.json
        fi
    fi
fi

# Copy also dictionaries (we will need this for standalone)
cp $FEATURES_FOLDER/dict.*.txt $checkpoints_dir/

echo "fairseq-train $FAIRSEQ_TRAIN_ARGS --seed $seed --save-dir $checkpoints_dir"
fairseq-train $FAIRSEQ_TRAIN_ARGS \
    --seed $seed \
    --save-dir $checkpoints_dir 

# Debug version
#kernprof -l train.py $FAIRSEQ_GENERATE_ARGS 
#python -m line_profiler train.py.lprof
