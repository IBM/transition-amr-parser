#!/bin/bash

set -o errexit
set -o pipefail


rootdir=/dccstor/jzhou1/work/EXP

##### sanity check: all the data configurations cover all the experiments
function config_data_coverage {

config_data_dir=run_tp/config_data
config_data_list=(${config_data_dir}/*)

for config_data in "${config_data_list[@]}"; do
    data_tag="$(basename $config_data | sed 's@config_data_\(.*\)\.sh@\1@g')"
    # check if the any file with the name pattern exists (otherwise ls will return an error)
    if test -n "$(find $rootdir -maxdepth 1 -name exp_${data_tag}* -print -quit)"
    then
        ls ${rootdir}/exp_${data_tag}* -dl
    fi
done

}

n1=$(config_data_coverage | wc -l)
n2=$(ls $rootdir/exp* -dl | wc -l)

echo [sanity check]
echo "data configuration covers this number of experiment folders: $n1"
echo "total number of experiment folders (including a debugging one): $n2"
if (( $n1 == $n2 - 1 )); then
    echo [check passed]
else
    echo [check failed] && exit 1
fi

###################################


# if [[ ! -z $1 ]]; then
#     config_data=$1
# else
#     config_data=run_tp/config_data/config_data_o5_no-mw_roberta-large-top24.sh
# fi

# set -o nounset

config_data_dir=run_tp/config_data
config_data_list=(${config_data_dir}/*)

for config_data in "${config_data_list[@]}"; do

    data_tag="$(basename $config_data | sed 's@config_data_\(.*\)\.sh@\1@g')"
    dirs=(${rootdir}/exp_${data_tag}*)

    for f in "${dirs[@]}"; do
    
        # check if any file with the data config exists
        [[ ! -d $f ]] && echo "$f not existing" && break
        
        # copying data configuration file into experiment folder
        echo "copying [$config_data] into [$f]"
        cp $config_data $f/
        
        # for initial run to be safe
        # break

    done

done

