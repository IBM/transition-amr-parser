set -o errexit 
set -o pipefail
. set_environment.sh
set -o nounset 

# Delete previous runs is exist
rm -Rf DATA/wiki25/*
rm -Rf data/wiki25*
rm -Rf EXP/exp_wiki25.*

# simulate completed corpora extraction and alignment
bash tests/create_wiki25_mockup.sh

# Run local test
# train
bash run_tp/run_model_action-pointer.sh config_files/wiki25.sh 42

# test
bash run_tp/jbsub_run_eval.sh config_files/wiki25.sh 42
