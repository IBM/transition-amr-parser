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
# set environment (needed for the python code below)
# NOTE: Old set_environment.sh forbids launching in login node.
. set_environment.sh
set -o nounset

# decode in paralel to training. ATTENTION: In that case we can not kill this
# script until first model appears
on_the_fly_decoding=true

# Load config
echo "[Configuration file:]"
echo $config
. $config

# Exit if we launch this directly from a computing node
if [[ "$HOSTNAME" =~ dccpc.* ]] || [[ "$HOSTNAME" =~ dccx[cn].* ]] || [[ "$HOSTNAME" =~ cccx[cn].* ]];then
    echo -e "\n$0 must be launched from a login node (submits its own jbsub calls)\n" 
    exit 1
fi

# Quick exits
# Data not extracted or aligned data not provided
if [ ! -f "$AMR_TRAIN_FILE_WIKI" ] && [ ! -f "$ALIGNED_FOLDER/train.txt" ];then
    echo -e "\nNeeds $AMR_TRAIN_FILE_WIKI or $ALIGNED_FOLDER/train.txt\n" 
    exit 1
fi

# Aligned data not provided, but alignment tools not installed
if [ ! -f "${ALIGNED_FOLDER}train.txt" ] && [ ! -f "preprocess/kevin/run.sh" ];then
    echo -e "\nNeeds ${ALIGNED_FOLDER}train.txt or installing aligner\n"
    exit 1
fi    

# Determine tools folder as the folder where this script is. This alloews its
# use when softlinked elsewhere
tools_folder=$(dirname $0)

# Ensure jbsub basename does not have forbidden symbols
jbsub_basename=$(echo $jbsub_basename | sed "s@[+]@_@g")

# create folder for each random seed and store a copy of the config there.
# Refer to that config on all posterio calls
for seed in $SEEDS;do

    # define seed and working dir
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
    jbsub -cores "1+1" -mem 50g -q x86_6h -require v100 \
          -name "$jbsub_tag" \
          -out $ORACLE_FOLDER/${jbsub_tag}-%J.stdout \
          -err $ORACLE_FOLDER/${jbsub_tag}-%J.stderr \
          /bin/bash run/aa_amr_actions.sh $config

    # train will wait for this to start
    prepro_depends="-depend $jbsub_tag"

else

    echo "skiping $ORACLE_FOLDER/.done"

    # resume from extracted
    prepro_depends=""

fi

# preprocessing
echo "[Preprocessing data:]"
if [[ (! -f $DATA_FOLDER/.done) || (! -f $EMB_FOLDER/.done) ]]; then

    # Run preprocessing
    jbsub_tag="fe-${jbsub_basename}-$$"
    jbsub -cores "1+1" -mem 50g -q x86_6h -require v100 \
          -name "$jbsub_tag" \
          $prepro_depends \
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
for seed in $SEEDS;do

    # define seed and working dir
    checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"

    if [ ! -f "$checkpoints_dir/checkpoint${MAX_EPOCH}.pt" ];then

        mkdir -p "$checkpoints_dir"

        # run new training
        jbsub_tag="tr-${jbsub_basename}-s${seed}-$$"
        jbsub -cores 1+1 -mem 50g -q x86_24h -require v100 \
              -name "$jbsub_tag" \
              $train_depends \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash run/ac_train.sh $config "$seed"

        # testing will wait for training to be finished
        test_depends="-depend $jbsub_tag"

    else

        echo "skiping $checkpoints_dir/.done"

        # resume from trained model, start test directly
        test_depends=""

    fi

    if [ "$on_the_fly_decoding" = false ];then

        # test all available checkpoints and link the best model on dev too
        jbsub_tag="tdec-${jbsub_basename}-s${seed}-$$"
        jbsub -cores 1+1 -mem 50g -q x86_24h -require v100 \
              -name "$jbsub_tag" \
              $test_depends \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash run/run_model_eval.sh $config "$seed"

    fi

done

# If we are doing on the fly decoding, we need to wait in this script until all
# seeds have produced a model to launch the testers
if [ "$on_the_fly_decoding" = true ];then
    for seed in $SEEDS;do

        # wait until first model is available
        # TODO: python run/status.py -c $config --seed $seed --wait-checkpoint-ready-to-eval
        while [ "$(python run/status.py -c $config --seed $seed --list-checkpoints-ready-to-eval --remove)" == "" ];do
            clear
            echo "Waiting for checkpoint $EVAL_INIT_EPOCH from seed $seed to be evaluated"
            echo ""
            echo "If you stop this evaluation wont be carried out!. set on_the_fly_decoding = false to avoid this"
            python run/status.py -c $config --seed $seed
            sleep 10
        done

        # test all available checkpoints and link the best model on dev too
        jbsub_tag="tdec-${jbsub_basename}-s${seed}-$$"
        jbsub -cores 1+1 -mem 50g -q x86_24h -require v100 \
              -name "$jbsub_tag" \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash run/run_model_eval.sh $config "$seed"

    done
fi

# inform of progress 
while true;do
    clear
    echo "Status of experiment $config"
    echo ""
    echo "(you can close this any time, use python run/status.py -c $config to check status)"
    python run/status.py -c $config
    sleep 10
done
