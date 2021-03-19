#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP
rootdir=EXP

exp_dirs=($rootdir/exp_o*)


for exp_dir in "${exp_dirs[@]}"; do

    # echo "$exp_dir"

    if [[ $exp_dir == *"dec-sep-emb"* ]]; then
        if [[ $exp_dir == *"bart-dec-emb-in"* ]] || [[ $exp_dir == *"bart-dec-emb-comp-pred"* ]] || [[ $exp_dir == *"bart-init-dec-emb"* ]]; then
            echo "$exp_dir"
            mv $exp_dir ${exp_dir}_bad-vocab-map
        else
            :
        fi
    else
        echo "$exp_dir"
        mv $exp_dir ${exp_dir}_bad-vocab-map
    fi

    # if [[ $exp_dir == *"bart-init-dec-emb"* ]]; then
    #     echo "$exp_dir"
    # fi

done
