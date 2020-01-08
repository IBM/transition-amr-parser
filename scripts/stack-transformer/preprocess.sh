set -o errexit 
set -o pipefail
#set -o nounset 

# load config
[ -z "$1" ] && echo "$0 config.sh" && exit 1
config=$1
. "$config"

# setup environment
. set_environment.sh

# stage-1: Feature Extraction

# feature extraction
# extract data
echo "fairseq-preprocess $fairseq_preprocess_args"
fairseq-preprocess $fairseq_preprocess_args 

# relink dev as test 
bash scripts/stack-transformer/replace_test_by_dev.sh $checkpoints_folder
