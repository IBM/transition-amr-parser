set -o errexit 
set -o pipefail
#set -o nounset 

# Argument handling
config=$1
seed=$2

# Load config
. "$config"

# setup environment
. set_environment.sh

# this is given by calling script to iterate over seeds
checkpoints_dir="${checkpoints_dir_root}-seed${seed}/"

[ ! -d "$checkpoints_dir" ] && \
    mkdir -p "$checkpoints_dir"

# If rules were used, copy them to model folder
if [ -f "$extracted_oracle_folder/${data_set}_extracted/train.rules.json" ];then
    cp $extracted_oracle_folder/${data_set}_extracted/train.rules.json $checkpoints_dir
fi

echo "fairseq-train $fairseq_train_args --seed $seed --save-dir $checkpoints_dir"
fairseq-train $fairseq_train_args \
    --seed $seed \
    --save-dir $checkpoints_dir 
