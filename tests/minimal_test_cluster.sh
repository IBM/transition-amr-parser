set -o errexit 
set -o pipefail
set -o nounset 

# Delete previous runs is exist
rm -Rf DATA/wiki25/*

# simulate completed corpora extraction and alignment
bash tests/create_wiki25_mockup.sh

# Run test in LSF cluster
bash run/lsf/run_experiment.sh configs/wiki25.sh  
