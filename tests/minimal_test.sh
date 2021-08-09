set -o errexit 
set -o pipefail
set -o nounset 

# Delete previous runs is exist
rm -Rf DATA/wiki25/*
# simulate completed corpora extraction and alignment
bash tests/create_wiki25_mockup.sh

# replace code above with less restrictive deletion
# rm -R -f DATA/wiki25/embeddings
# rm -R -f DATA/wiki25/features
# rm -R -f DATA/wiki25/oracles
# rm -R -f DATA/wiki25/models

# config=configs/wiki25-neur-al-sampling.sh
config=configs/wiki25.sh  
bash run/run_experiment.sh $config

# check if final result is there
. $config

if [ -f "${MODEL_FOLDER}-seed42/beam10/valid_${DECODING_CHECKPOINT}.wiki.smatch" ];then
    printf "\n[\033[92mOK\033[0m] $0\n"
else
    printf "\n[\033[91mFAILED\033[0m] $0\n"
fi
