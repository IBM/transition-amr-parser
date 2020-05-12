set -o errexit
set -o pipefail
[ -z "$1" ] && echo -e "\ne.g. bash $0 /path/to/config.sh\n" && exit 1
config=$1
set -o nounset

# load config 
. $config

# Determine tools folder as the folder where this script is. This alloews its
# use when softlinked elsewhere
tools_folder=$(dirname $0)

# create folder for each random seed and store a copy of the config there.
# Refer to that config on all posterio calls
for index in $(seq $NUM_SEEDS);do

    # define seed and working dir
    seed=$((41 + $index))
    checkpoints_dir="${CHECKPOINTS_DIR_ROOT}-seed${seed}/"

    # create repo
    mkdir -p $checkpoints_dir   

    # copy config and store in model folder
    cp $config $checkpoints_dir/config.sh

done

# preprocessing
echo "stage-1: Preprocess"
if [ ! -f "$features_folder/train.en-actions.actions.bin" ];then

    mkdir -p "$ORACLE_FOLDER"
    mkdir -p "$features_folder"

    bash $tools_folder/preprocess.sh $checkpoints_dir/config.sh

fi

echo "stage-2/3: Training/Testing (multiple seeds)"
# Launch one training instance per seed
for index in $(seq $NUM_SEEDS);do

    # define seed and working dir
    seed=$((41 + $index))
    checkpoints_dir="${CHECKPOINTS_DIR_ROOT}-seed${seed}/"

    if [ ! -f "$checkpoints_dir/checkpoint${MAX_EPOCH}.pt" ];then

        mkdir -p "$checkpoints_dir"

        # run new training
        bash $tools_folder/train.sh $checkpoints_dir/config.sh "$seed"

    fi

    # test all available checkpoints and rank them
    bash $tools_folder/epoch_tester.sh $checkpoints_dir/

done
