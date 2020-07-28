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

# this is given by calling script to iterate over seeds
checkpoints_dir="${CHECKPOINTS_DIR_ROOT}-seed${seed}/"

[ ! -d "$checkpoints_dir" ] && \
    mkdir -p "$checkpoints_dir"

# If rules were used, copy them to model folder
if [ -f "$ORACLE_FOLDER/train.rules.json" ];then
    echo "cp $ORACLE_FOLDER/train.rules.json $checkpoints_dir"
    cp $ORACLE_FOLDER/train.rules.json $checkpoints_dir
fi

# Copy entity_rules.json from oracle, created using train file
if [ "$ENTITY_RULES" == "" ]; then
    if [ ! -f "$ORACLE_FOLDER/entity_rules.json" ];then
	cp $ORACLE_FOLDER/entity_rules.json $checkpoints_dir
    fi
else
    cp $ENTITY_RULES $checkpoints_dir
fi

# Copy also dictionaries (we will need this for standalone)
cp $features_folder/dict.*.txt $checkpoints_dir/

echo "fairseq-train $FAIRSEQ_TRAIN_ARGS --seed $seed --save-dir $checkpoints_dir"
fairseq-train $FAIRSEQ_TRAIN_ARGS \
    --seed $seed \
    --save-dir $checkpoints_dir 

# Debug version
#kernprof -l train.py $FAIRSEQ_GENERATE_ARGS 
#python -m line_profiler train.py.lprof
