set -o nounset
set -o pipefail 
set -o errexit 

[ ! -d scripts/ ] && echo "Call as scripts/$(basename $0)" && exit 1
. scripts/local_variables.sh

[ ! -d data/ ] && mkdir data/

# create oracle data
python data_oracle.py \
    --in-amr $dev_file \
    --out-amr data/dev.oracle.amr \
    --out-sentences data/dev.tokens \
    --out-actions data/dev.actions \

# parse a sentence step by step
python parse.py \
    --in-sentences data/dev.tokens \
    --in-actions data/dev.actions \
    --offset 0 \
    --step-by-step
