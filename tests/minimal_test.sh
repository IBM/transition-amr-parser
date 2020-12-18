set -o errexit 
set -o pipefail
set -o nounset 

# simulate completed corpora extraction and alignment
. tests/create_wiki25_mockup.sh

# Run local test
bash run/run_experiment.sh configs/wiki25.sh  
