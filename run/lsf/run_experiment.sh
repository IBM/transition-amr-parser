#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\ne.g. bash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
config=$1
if [ -z "$2" ];then
    # identify experiment by the repository tag
    jbsub_basename="$(basename $config | sed 's@\.sh$@@')"
else
    # identify experiment by given tag
    jbsub_basename=$2
fi
set -o nounset

NUM_SEEDS=1

# Load config
echo "[Configuration file:]"
echo $config
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
    checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"

    # create repo
    mkdir -p $checkpoints_dir   

    # copy config and store in model folder
    cp $config $checkpoints_dir/config.sh

done

# preprocessing
echo "[Building oracle actions:]"
if [ ! -f "$ORACLE_FOLDER/.done" ];then

    # Run preprocessing
    jbsub_tag="or-${jbsub_basename}-$$"
    jbsub -cores "1+1" -mem 50g -q x86_6h -require k80 \
          -name "$jbsub_tag" \
          -out $ORACLE_FOLDER/${jbsub_tag}-%J.stdout \
          -err $ORACLE_FOLDER/${jbsub_tag}-%J.stderr \
          /bin/bash run/aa_amr_actions.sh $config

    # train will wait for this to start
    train_depends="-depend $jbsub_tag"

else

    echo "skiping $ORACLE_FOLDER/.done"

    # resume from extracted
    train_depends=""

fi

# preprocessing
echo "[Preprocessing data:]"
if [[ (! -f $DATA_FOLDER/.done) && (! -f $EMB_FOLDER/.done) ]]; then

    # Run preprocessing
    jbsub_tag="fe-${jbsub_basename}-$$"
    jbsub -cores "1+1" -mem 50g -q x86_6h -require k80 \
          -name "$jbsub_tag" \
          -out $ORACLE_FOLDER/${jbsub_tag}-%J.stdout \
          -err $ORACLE_FOLDER/${jbsub_tag}-%J.stderr \
          /bin/bash run/ab_preprocess.sh $config

    # train will wait for this to start
    train_depends="-depend $jbsub_tag"

else

    echo "skiping $EMB_FOLDER/.done"
    echo "skiping $DATA_FOLDER/.done"

    # resume from extracted
    train_depends=""

fi

echo "[Training:]"
# Launch one training instance per seed
for index in $(seq $NUM_SEEDS);do

    # define seed and working dir
    seed=$((41 + $index))
    checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"

    if [ ! -f "$checkpoints_dir/checkpoint${MAX_EPOCH}.pt" ];then

        mkdir -p "$checkpoints_dir"

        # run new training
        jbsub_tag="tr-${jbsub_basename}-s${seed}-$$"
        jbsub -cores 1+1 -mem 50g -q ppc_24h -require v100 \
              -name "$jbsub_tag" \
              $train_depends \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash run/ac_train.sh $config "$seed"

        # testing will wait for this name to start
        test_depends="-depend $jbsub_tag"

    else

        echo "skiping $checkpoints_dir/.done"

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
          /bin/bash run/epoch_tester.sh $checkpoints_dir/
    test_depends="-depend $jbsub_tag"

    # beam test 
    jbsub_tag="beam-${jbsub_basename}-s${seed}-$$"
    jbsub -cores 1+1 -mem 50g -q x86_6h -require v100 \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash run/ad_test.sh \
            $checkpoints_dir/$DECODING_CHECKPOINT \
            dev \
            $BEAM_SIZE

done
