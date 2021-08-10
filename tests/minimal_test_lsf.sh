set -o errexit 
set -o pipefail
set -o nounset 

config=configs/wiki25.sh  

# Delete previous runs is exist
rm -Rf DATA/wiki25/*

# simulate completed corpora extraction and alignment
bash tests/create_wiki25_mockup.sh

# Run local test
bash run/lsf/run_experiment.sh $config  

# check if final result is there
. $config

if [ -f "${MODEL_FOLDER}-seed42/beam10/valid_${DECODING_CHECKPOINT}.wiki.smatch" ];then
    printf "\n[\033[92mOK\033[0m] $0\n"
else
    printf "\n[\033[91mFAILED\033[0m] $0\n"
fi
