#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP
rootdir=EXP

# rename previous experiments with bad vocabulary mapping to bart vocabulary
# exp_dirs=($rootdir/exp_o*)

# for exp_dir in "${exp_dirs[@]}"; do

#     # echo "$exp_dir"

#     if [[ $exp_dir == *"dec-sep-emb"* ]]; then
#         if [[ $exp_dir == *"bart-dec-emb-in"* ]] || [[ $exp_dir == *"bart-dec-emb-comp-pred"* ]] || [[ $exp_dir == *"bart-init-dec-emb"* ]]; then
#             echo "$exp_dir"
#             mv $exp_dir ${exp_dir}_bad-vocab-map
#         else
#             :
#         fi
#     else
#         echo "$exp_dir"
#         mv $exp_dir ${exp_dir}_bad-vocab-map
#     fi

#     # if [[ $exp_dir == *"bart-init-dec-emb"* ]]; then
#     #     echo "$exp_dir"
#     # fi

# done


# rename previous experiments with o10 to o9.0
otag1=o10
otag2=o9.0

exp_dirs=($rootdir/exp_${otag1}_*)

for exp_dir in "${exp_dirs[@]}"; do

    # echo "$exp_dir"

    new=$(echo "$exp_dir" | sed "s/exp_${otag1}_/exp_${otag2}_/g")

    mv $exp_dir $new
    echo "$exp_dir --> $new"

done

mv $rootdir/data/${otag1}_act-states $rootdir/data/${otag2}_act-states
echo "$rootdir/data/${otag1}_act-states --> $rootdir/data/${otag2}_act-states"
