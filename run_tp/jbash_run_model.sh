#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh

##### data config
if [ -z "$1" ]; then
    config_model=config_model_action-pointer.sh
else
    config_model=$1
fi

# to detect any reuse of $1 which is shared across all sourced files and may cause error
set -o nounset

dir=$(dirname $0)

# load model configuration so that the command knows where to log
. $dir/$config_model

# this is set in the above file sourced
# set -o nounset

##### run the job directly (e.g. in interactive mode)

# this is necessary for output redirected to file
mkdir -p $MODEL_FOLDER

/bin/bash $dir/run_model_action-pointer.sh $config_model #|& tee $MODEL_FOLDER/run.log

# bash_x86_12h_v100 $dir/run_model_action-pointer.sh $config_model |& tee $MODEL_FOLDER/run.log



