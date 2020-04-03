set -o errexit
set -o pipefail
[ -z "$1" ] && echo -e "\ne.g. bash $0 /path/to/config.sh\n" && exit 1
config=$1
if [ -z "$2" ];then
    # identify experiment by the repository tag
    repo_tag="$(basename $(pwd) | sed 's@.*\.@@')"
else
    # identify experiment by given tag
    repo_tag=$2
fi
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

    # run preprocessing
    jbsub_tag="pr-${repo_tag}-$$"
    jbsub -cores "1+1" \
          -mem 50g \
          -q "$PREPRO_QUEUE" \
          -require "$PREPRO_GPU_TYPE" \
          -name "$jbsub_tag" \
          -out $features_folder/${jbsub_tag}-%J.stdout \
          -err $features_folder/${jbsub_tag}-%J.stderr \
          /bin/bash $tools_folder/preprocess.sh $checkpoints_dir/config.sh

    # train will wait for this to start
    train_depends="-depend $jbsub_tag"

else

    # resume from extracted
    train_depends=""

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
        jbsub_tag="tr-${repo_tag}-s${seed}-$$"
        jbsub -cores 1+1 \
              -mem 50g \
              -q "$TRAIN_QUEUE" \
              -require "$TRAIN_GPU_TYPE" \
              -name "$jbsub_tag" \
              $train_depends \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash $tools_folder/train.sh $checkpoints_dir/config.sh "$seed"

        # testing will wait for this name to start
        test_depends="-depend $jbsub_tag"

    else

        # resume from trained model, start test directly
        test_depends=""

    fi

    # test all available checkpoints and rank them
    jbsub_tag="tdec-${repo_tag}-$$"
    jbsub -cores 1+1 \
          -mem 50g \
          -q "$TEST_QUEUE" \
          -require "$TEST_GPU_TYPE" \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash $tools_folder/epoch_tester.sh $checkpoints_dir/

done
