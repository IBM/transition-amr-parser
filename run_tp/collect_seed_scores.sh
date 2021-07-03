#!/bin/bash

set -o errexit
set -o pipefail

# setup environment
. set_environment.sh

[ -z $1 ] && echo "usage: bash collect_seed_scores.sh <exp_dir>" && exit 1
set -o nounset

exp_dir=$1

echo -e "\n[Summarized results across different random seeds under experiments:]"
echo $exp_dir
echo

score_name=wiki.smatch
if [[ $exp_dir == *"amr1"* ]]; then
    score_name=smatch
fi

# data_sets=("dev")
# data_sets=("test")
data_sets=("dev" "test")

for ((i=0;i<${#data_sets[@]};++i)); do
    if [[ ${data_sets[i]} == "dev" ]]; then
        data_sets[$i]=valid
    fi
done

epoch=120
seeds="42 0 315"

models="last wiki-smatch_best1 wiki-smatch_top3-avg wiki-smatch_top5-avg"
beam_sizes="1 5 10"

# models="wiki-smatch_top5-avg"
# beam_sizes="10"

# the following will join the string list to a single string;
# this is in constrast with: ${data_sets[*]}, or ${data_sets[@]}, or "${data_sets[@]}"
for data in "${data_sets[*]}"; do

    python run_tp/collect_seed_scores.py \
        $exp_dir \
        --score_name $score_name \
        --data_sets $data \
        --ndigits 1 \
        --models $models \
        --beam_sizes $beam_sizes \
        --epoch $epoch \
        --seeds $seeds

done