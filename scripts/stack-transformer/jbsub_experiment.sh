set -o errexit
set -o pipefail
[ -z "$1" ] && echo -e "\ne.g. bash $0 /path/to/config.sh\n" && exit 1
config=$1
if [ -z "$2" ];then
    # identify experiment by the repository tag
    jbsub_basename="$(basename $config | sed 's@\.sh$@@')"
else
    # identify experiment by given tag
    jbsub_basename=$2
fi
set -o nounset

# load config 
. $config

# Determine tools folder as the folder where this script is. This alloews its
# use when softlinked elsewhere
tools_folder=$(dirname $0)

# Ensure jbsub basename does not have forbidden symbols
jbsub_basename=$(echo $jbsub_basename | sed "s@[+]@_@g")

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
if [ ! -f "$FEATURES_FOLDER/train.en-actions.actions.bin" ];then

    mkdir -p "$ORACLE_FOLDER"
    mkdir -p "$FEATURES_FOLDER"

    # Run preprocessing
    jbsub_tag="pr-${jbsub_basename}-$$"
    jbsub -cores "1+1" -mem 50g -q x86_6h -require k80 \
          -name "$jbsub_tag" \
          -out $FEATURES_FOLDER/${jbsub_tag}-%J.stdout \
          -err $FEATURES_FOLDER/${jbsub_tag}-%J.stderr \
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
        jbsub_tag="tr-${jbsub_basename}-s${seed}-$$"
        jbsub -cores 1+1 -mem 50g -q ppc_24h -require v100 \
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

    # test all available checkpoints and link the best model on dev to
    # $CHECKPOINT
    jbsub_tag="tdec-${jbsub_basename}-s${seed}-$$"
    jbsub -cores 1+1 -mem 50g -q x86_6h -require v100 \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash $tools_folder/epoch_tester.sh $checkpoints_dir/
    test_depends="-depend $jbsub_tag"

    # beam test 
    jbsub_tag="beam-${jbsub_basename}-s${seed}-$$"
    jbsub -cores 1+1 -mem 50g -q x86_6h -require v100 \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash $tools_folder/beam_test.sh $checkpoints_dir/

done
