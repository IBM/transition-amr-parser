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

# wait until first checkpoint is available for any of the seeds. 
# Clean-up checkpoints and inform of status in the meanwhile
python run/status.py -c $config \
    --wait-checkpoint-ready-to-eval --clear --remove

for seed in $SEEDS;do

    checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"

    # test all available checkpoints and link the best model on dev too
    jbsub_tag="tdec-${jbsub_basename}-s${seed}-$$"
    jbsub -cores 1+1 -mem 50g -q x86_24h -require v100 \
          -name "$jbsub_tag" \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash run/run_model_eval.sh $config "$seed"

done

# wait until final models has been evaluated 
# NOTE checkpoints are cleaned-up by run_model_eval.sh
python run/status.py -c $config --wait-finished --clear
