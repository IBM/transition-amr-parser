#!/bin/bash
set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
HELP="$0 <checkpoint> <tokenized sentences file>"
[ "$#" -lt 2 ] && echo "$HELP" && exit 1
train_amr=$1
input_file=$2
set -o nounset

amr-parse \
    --in-checkpoint $in_checkpoint \
    --in-tokenized-sentences $input_file \
    --out-amr $(basename $input_file).amr
