#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
. ~/jbfcn.sh    # short form command to submit bash jobs
set -o nounset

dir=$(dirname $0)
config=config_dev.sh


# we should always call from one level up

# bash $dir/aa_amr_actions.sh $config

# bash $dir/amr_oracle_smatch.sh $config

# bash_x86 $dir/ab_preprocess.sh $config

bash $dir/ac_train.sh $config
# bash_x86_12h_v100 $dir/ac_train.sh $config

############# change back the amr rule for LA(root) LA(root)!
# bash_x86 $dir/ad_test.sh $config dev
# bash_x86 $dir/ad_test.sh $config test