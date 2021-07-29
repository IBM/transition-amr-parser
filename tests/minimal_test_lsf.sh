set -o errexit 
set -o pipefail
set -o nounset 

# Delete previous runs is exist
rm -Rf DATA/wiki25/*

# simulate completed corpora extraction and alignment
bash tests/create_wiki25_mockup.sh

# Run local test
bash run/lsf/run_experiment.sh configs/wiki25.sh  
