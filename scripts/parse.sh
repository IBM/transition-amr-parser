set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <checkpoint> <tokenized sents> <out_amr>"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1
[ -z $3 ] && echo "$HELP" && exit 1
checkpoint=$1
tokenized_sentences=$2
out_amr=$3
set -o nounset

amr-parse \
    --in-checkpoint $checkpoint \
    --in-tokenized-sentences $tokenized_sentences \
    --out-amr $out_amr
