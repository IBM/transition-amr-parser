set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <checkpoint> <amr file>"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1
checkpoint=$1
amr_file=$2
set -o nounset

# Get tokens
mkdir -p DATA.test
tokens_file=DATA.test/$(basename $amr_file).tokens
out_amr=DATA.test/$(basename $checkpoint)_$(basename $amr_file).amr
rm -f $out_amr
grep '::tok' $amr_file | sed 's@# ::tok @@' > $tokens_file

# Parse them
#amr-parse \
pyinstrument transition_amr_parser/parse.py \
    --in-checkpoint $checkpoint \
    --in-tokenized-sentences $tokens_file \
    --out-amr $out_amr

# Smatch
smatch.py -r 10 --significant 4 -f $amr_file $out_amr
