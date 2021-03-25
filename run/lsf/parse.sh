#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
# First argument must be checkpoint
HELP="\nbash $0 <checkpoint> <tokenized_sentences< [-s <max_split_size>]\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ -z "$2" ] && echo -e "$HELP" && exit 1
first_path=$(echo $1 | sed 's@:.*@@g')
[ ! -f "$first_path" ] && "Missing $1" && exit 1
checkpoint=$1
tokenized_sentences=$2
# process the rest with argument parser
max_split_size=2000
shift 
shift 
while [ "$#" -gt 0 ]; do
  case "$1" in
    -s) max_split_size="$2"; shift 2;;
    *) echo "unrecognized argument: $1"; exit 1;;
  esac
done

# Split files
split -l $max_split_size $tokenized_sentences ${tokenized_sentences}.split_

# Launch multiple decodings jobs
for split in $(ls ${tokenized_sentences}.split_*);do
    jbsub -cores 1+1 -mem 50g -q x86_6h -require v100 \
          -name $(basename $split) \
          -out $(dirname $split)%J.stdout \
          -err $(dirname $split)/%J.stderr \
          /bin/bash run/parse.sh $checkpoint $split ${split}.amr
done
