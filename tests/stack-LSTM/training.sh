#
# Small (non realistic) train iteration to check it works
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
    -A $train_file \
    -a $dev_file \
    -B $train_bert  \
    -b $dev_bert \
    --save_model models/$name \
    --desc "$name" \
    --name model \
    --no_chars \
    --no_bert \
    --cores $num_cores \
    --batch $batch_size \
    --lr $lr \
