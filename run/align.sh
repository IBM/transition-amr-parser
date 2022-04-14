set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <checkpoint> <in_amr> <out_amr>"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1
[ -z $3 ] && echo "$HELP" && exit 1
checkpoint=$1
in_amr=$2
out_amr=$3
set -o nounset

amr-parse --in-checkpoint $checkpoint --in-amr $in_amr --out-amr $out_amr --batch-size 512 --roberta-batch-size 512
