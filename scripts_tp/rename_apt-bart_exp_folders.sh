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


# rename previous experiments with o10 to o9.0 -> o9.1 -> o9.2
# o9.0: initial o10
# o9.1: initial o10 + arc order fix, but still with some little issue
# o9.2: before the fix on connect_graph(), which would still make gold actions include :rel
otag1=o10
otag2=o9.2

exp_dirs=($rootdir/exp*${otag1}_*)

for exp_dir in "${exp_dirs[@]}"; do

    # echo "$exp_dir"

    new=$(echo "$exp_dir" | sed "s/_${otag1}_/_${otag2}_/g")

    mv $exp_dir $new
    echo "$exp_dir --> $new"

done

data_dirs=($rootdir/data/*${otag1}_act-states)

for data_dir in "${data_dirs[@]}"; do

    # echo "$data_dir"

    new=$(echo "$data_dir" | sed "s/${otag1}_/${otag2}_/g")

    mv $data_dir $new
    echo "$data_dir --> $new"

done
