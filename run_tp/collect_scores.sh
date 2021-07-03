#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\ntest_all_checkpoints.sh <checkpoints_folder>\n" && \
    exit 1
set -o nounset

score_name=wiki.smatch
if [[ $checkpoints_folder == *"amr1"* ]]; then
    score_name=smatch
fi

# data_sets=("dev")
# data_sets=("test")
data_sets=("dev" "test")

# or (loop over a list of strings)
# data_sets="dev test"
# for data in $data_sets; do
#     echo $data
# done

# for data in ${data_sets[@]}; do

#     if [[ $data == "dev" ]]; then
#         data=valid
#     fi

# done

for ((i=0;i<${#data_sets[@]};++i)); do
    if [[ ${data_sets[i]} == "dev" ]]; then
        data_sets[$i]=valid
    fi
done



# the following will join the string list to a single string;
# this is in constrast with: ${data_sets[*]}, or ${data_sets[@]}, or "${data_sets[@]}"
for data in "${data_sets[*]}"; do

    python run_tp/collect_scores.py \
           $checkpoints_folder \
           --score_name $score_name \
           --data_sets $data \
           --ndigits 2

done
