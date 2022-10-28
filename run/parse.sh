set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <checkpoint> <tokenized sents> <out_amr> [--tokenize]"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1
[ -z $3 ] && echo "$HELP" && exit 1
checkpoint=$1
tokenized_sentences=$2
out_amr=$3

tokenize=""
shift 3
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tokenize) tokenize="--tokenize"; shift 1;;
    *) echo "unrecognized argument: $1"; exit 1;;
  esac
done

amr-parse \
    --in-checkpoint $checkpoint \
    --in-tokenized-sentences $tokenized_sentences \
    --out-amr $out_amr \
    $tokenize
