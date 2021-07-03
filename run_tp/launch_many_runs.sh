#!/bin/bash

set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset


# config_list=(run_tp/config_model_data-amr1_depfix_o5_no-mw_vmask1/*)

# for config_model in ${config_list[@]}; do
#     # if [[ $(basename $config_model) != "config_model_action-pointer_data-amr1_depfix_o5-no-mw_ptr-layer6-head1_cam-layerall-head1_tis.sh" ]]; then
#     echo $config_model
#     bash run_tp/jbsub_run_model-eval_seeds.sh $config_model
#     # fi
# done



# config_dir=run_tp/config_model_data_depfix_o5_no-mw_vmask1

# bash run_tp/jbsub_run_model-eval_seeds.sh $config_dir/config_model_action-pointer_data_depfix_o5-no-mw_ptr-layer6-head1_cam-layerall-head2-a-buf_shiftpos1.sh

config_list=(run_tp/config_model_data_depfix_o5_no-mw_vmask1/*)
# config_list=(run_tp/config_model_data_depfix_o5_no-mw_vmask1_pmask1/*)

for config_model in ${config_list[@]}; do
    echo $config_model
    bash run_tp/jbsub_run_model-eval_seeds.sh $config_model
done