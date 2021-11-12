#!/bin/bash
set -o pipefail 
set -o errexit 
# load local variables used below
. set_environment.sh
HELP="$0 <checkpoint> <tokenized sentences file> <out amr>"
[ "$#" -lt 3 ] && echo "$HELP" && exit 1
checkpoint=$1
input_file=$2
output_amr=$3
set -o nounset

amr-parse \
    --in-checkpoint $checkpoint \
    --in-tokenized-sentences $input_file \
    --out-amr $output_amr \
    --roberta-cache-path DATA/bart.large \
    --batch-size 128 \
    --roberta-batch-size 1
