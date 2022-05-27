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

# Load config
echo "[Configuration file:]"
echo $config
. $config

# MANUAL OVERRIDE !!
# BEAM_SIZE=1
# DECODING_CHECKPOINT=checkpoint_wiki.smatch_best1.pt

# Running test announcement
printf "\n\033[93mWARNING\033[0m: Everytime you look at the test set, your corpus dies a little (by corpus overfitting)\n\n" 
echo -e " \nbash run/ad_test.sh ${MODEL_FOLDER}-seed${SEEDS}/$DECODING_CHECKPOINT -b $BEAM_SIZE -s test\n"
read -p "Do you wish to continue? Y/[N]" answer
[ "$answer" != "Y" ] && exit 1

# Exit if we launch this directly from a computing node
if [[ "$HOSTNAME" =~ dccpc.* ]] || [[ "$HOSTNAME" =~ dccx[cn].* ]] || [[ "$HOSTNAME" =~ cccx[cn].* ]];then
    echo -e "\n$0 must be launched from a login node (submits its own jbsub calls)\n" 
    exit 1
fi

for seed in $SEEDS;do

    # define seed and working dir
    checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"

    # test all available checkpoints and link the best model on dev too
    jbsub_tag="fdec-${jbsub_basename}-s${seed}-$$"
    jbsub -cores 1+1 -mem 50g -q x86_24h -require v100 \
          -name "$jbsub_tag" \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash run/test.sh ${checkpoints_dir}/$DECODING_CHECKPOINT \
            -b $BEAM_SIZE \
            -s test

done
