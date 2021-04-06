set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <checkpoint> <amr file> [results_basename]"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1
checkpoint=$1
amr_file=$2
results_basename=$3
[ -z "$results_basename" ] && results_basename=""
set -o nounset

# Get tokens
mkdir -p DATA.test
tokens_file=DATA.test/$(basename $amr_file).tokens
first_path=$(echo $checkpoint | sed 's@:.*@@g')
if [ "$results_basename" == "" ];then
    out_amr=DATA.test/$(basename $first_path)_$(basename $amr_file).amr
else
    out_amr=$results_basename.amr
fi
rm -f $out_amr
grep '::tok' $amr_file | sed 's@# ::tok @@' > $tokens_file

# Parse them
# pyinstrument transition_amr_parser/parse.py \
amr-parse \
    --in-checkpoint $checkpoint \
    --in-tokenized-sentences $tokens_file \
    --out-amr $out_amr

# Smatch
smatch.py -r 10 --significant 4 -f $amr_file $out_amr
