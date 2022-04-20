#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config> <seed>\n"
# config file
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ ! -f "$1" ] && "Missing $1" && exit 1
config=$1
# random seed
[ -z "$2" ] && echo -e "$HELP" && exit 1
seed=$2

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# folder of the model seed
checkpoints_folder=${MODEL_FOLDER}-seed${seed}/
echo $checkpoints_folder

# Final run
[ ! -f "$checkpoints_folder/$DECODING_CHECKPOINT" ] \
    && echo -e "Missing $checkpoints_folder/$DECODING_CHECKPOINT" \
    && exit 1


#mkdir -p $checkpoints_folder/beam${BEAM_SIZE}/
#bash run/ad_test_smatch.sh $checkpoints_folder/$DECODING_CHECKPOINT -b $BEAM_SIZE

##### test on the best smatch model

if [[ -f $checkpoints_folder/checkpoint_wiki.smatch_best1.pt ]]; then

    test_model=checkpoint_wiki.smatch_best1.pt

    for beam_size in 1 5 10
    do
	mkdir -p $checkpoints_folder/beam${beam_size}/
        bash run/ad_test.sh $checkpoints_folder/$test_model -s dev -b $beam_size 
        bash run/ad_test.sh $checkpoints_folder/$test_model -s test -b $beam_size
    done

fi

##### Loop over existing averaged checkpoints                                                                  
for test_model in $(find $checkpoints_folder -iname 'checkpoint_wiki.smatch_top[0-9]*-avg.pt' | sort ); do

    echo -e "\n$test_model"
    echo "[Decoding and computing smatch:]"
    for beam_size in 1 5 10
    do
	bash run/ad_test.sh $test_model -s dev -b $beam_size
	bash run/ad_test.sh $test_model -s test -b $beam_size
    done

done
