set -o errexit
set -o pipefail
. set_environment.sh
# [ -z $1 ] && echo "$0 <amr_file (no wiki)>" && exit 1
# gold_amr=$1
[ -z "$1" ] && echo "$0 train (or dev, test) (wiki is not considered)" && exit 1
data=$1
set -o nounset

oracle_tag=o10

train_amr=amr_corpus/amr2.0/o5/jkaln.txt
dev_amr=amr_corpus/amr2.0/o5/dev.txt.removedWiki.noempty.JAMRaligned
test_amr=amr_corpus/amr2.0/o5/test.txt.removedWiki.noempty.JAMRaligned

debug_amr=DATA/debug.amr
train_ra_amr=amr_corpus/amr2.0/o5/jkaln.repairalignment.txt


if [[ $data == "train" ]]; then
    gold_amr=$train_amr
elif [[ $data == "dev" ]]; then
    gold_amr=$dev_amr
elif [[ $data == "test" ]]; then
    gold_amr=$test_amr
elif [[ $data == "debug" ]]; then
    gold_amr=$debug_amr
elif [[ $data == "train_ra" ]]; then
    gold_amr=$train_ra_amr
else
    echo "unsupported data split $data"
fi

oracle_folder=DATA/AMR2.0/oracles/$oracle_tag
mkdir -p $oracle_folder

python transition_amr_parser/o10_amr_machine.py \
    --in-aligned-amr $gold_amr \
    --out-machine-config $oracle_folder/machine_config.json \
    --out-actions $oracle_folder/${data}.actions \
    --out-tokens $oracle_folder/${data}.tokens \
    --absolute-stack-positions  \
    --out-stats-vocab $oracle_folder/${data}.actions.vocab \
    # --reduce-nodes all

python transition_amr_parser/o10_amr_machine.py \
    --in-machine-config $oracle_folder/machine_config.json \
    --in-tokens $oracle_folder/${data}.tokens \
    --in-actions $oracle_folder/${data}.actions \
    --out-amr $oracle_folder/${data}_oracle.amr

# Score
echo "compute Smatch"
smatch.py -r 10 --significant 4 -f $gold_amr $oracle_folder/${data}_oracle.amr > $oracle_folder/${data}_oracle.smatch
cat $oracle_folder/${data}_oracle.smatch
