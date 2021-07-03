#!/bin/bash

set -o errexit
set -o pipefail
# setup environment
. set_environment.sh

rootdir=/dccstor/jzhou1/work/EXP

# exp_dirs=($rootdir/exp_o*)
exp_dirs=($rootdir/exp_o5_no-mw*)

epoch_last=120

for exp_dir in "${exp_dirs[@]}"; do
    
    echo -e "\n[Decoding for all checkpoints under experiments:]"
    echo "$exp_dir"
    
    model_folders=($exp_dir/models*)
    
    for checkpoints_folder in "${model_folders[@]}"; do
        
        echo $checkpoints_folder

        jbsub_info=$(jbsub \
                 -cores 1+1 \
                 -mem 50g \
                 -q x86_6h \
                 -require v100 \
                 -name $0 \
                 -out $checkpoints_folder/logdec_topavg-%J.stdout \
                 -err $checkpoints_folder/logdec_topavg-%J.stderr \
                 /bin/bash run_tp/bc_test_model_avg.sh $checkpoints_folder \
                 | grep 'is submitted to queue')

        echo $jbsub_info
             
    done
    
    # to debug
    # break

done
