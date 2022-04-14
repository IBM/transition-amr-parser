#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
# First argument must be checkpoint
HELP="\nbash $0 <checkpoint> <in_amr> <out_amr> [-s <max_split_size>]\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
[ -z "$2" ] && echo -e "$HELP" && exit 1
[ -z "$3" ] && echo -e "$HELP" && exit 1
first_path=$(echo $1 | sed 's@:.*@@g')
[ ! -f "$first_path" ] && "Missing $1" && exit 1
checkpoint=$1
in_amr=$2
out_amr=$3
# process the rest with argument parser
max_split_size=2000
shift 
shift 
shift 
while [ "$#" -gt 0 ]; do
  case "$1" in
    -s) max_split_size="$2"; shift 2;;
    *) echo "unrecognized argument: $1"; exit 1;;
  esac
done

# splits folder
splits_folder=${out_amr}.${max_split_size}splits/
mkdir -p $splits_folder

# Split files
split_files=$(
    python scripts/split_amrs.py \
    $in_amr $max_split_size ${splits_folder}/in_split
)

# Launch multiple decodings jobs
for split in $split_files;do

    echo "bash run/align.sh $checkpoint $split ${split}.amr"

    if [ ! -f "${split}.amr" ];then

        jbsub -cores 1+1 -mem 50g -q x86_6h -require v100 \
              -name $(basename $split)-$$ \
              -out ${splits_folder}/align-%J-$$.stdout \
              -err ${splits_folder}/align-%J-$$.stderr \
              /bin/bash run/align.sh $checkpoint $split ${split}.amr
    
    fi

    exit
    
done
