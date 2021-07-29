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

#bash run/run_experiment.sh configs/wiki25.sh  
bash run/run_experiment.sh configs/wiki25-neur-al-sampling.sh
