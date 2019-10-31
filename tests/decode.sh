#
# TODO: Standalone call of parser
#
set -o errexit
set -o nounset
set -o pipefail 

# sanity checks
[ ! -d scripts ] && echo "to be run as bash scripts/train.sh" && exit 1

config=$1

# python modules

# load local variables used below
. $config 

# train model
amr-learn \
    --test_mode \
    -A $train_file \
    -a $dev_file \
    -B $train_bert  \
    -b $dev_bert \
    --load_model $trained_model \
    --desc "$name" \
    --name model \
    --no_chars \
    --cores $num_cores \
    --batch $batch_size \
    --lr $lr \
