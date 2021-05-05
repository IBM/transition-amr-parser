set -o errexit
set -o pipefail
. set_environment.sh
[ -z $1 ] && echo "$0 <amr_file (no wiki)>" && exit 1
gold_amr=$1
set -o nounset 

oracle_folder=DATA/AMR2.0/oracles/o10_pinitos/
mkdir -p $oracle_folder 

python transition_amr_parser/o10_amr_machine.py \
    --in-aligned-amr $gold_amr \
    --out-machine-config $oracle_folder/machine_config.json \
    --out-actions $oracle_folder/train.actions \
    --out-tokens $oracle_folder/train.tokens \
    --absolute-stack-positions  \
    # --reduce-nodes all

python transition_amr_parser/o10_amr_machine.py \
    --in-machine-config $oracle_folder/machine_config.json \
    --in-tokens tmp.tokens \
    --in-actions $oracle_folder/train.actions \
    --out-amr $oracle_folder/train_oracle.amr

# Score
#smatch.py -r 10 --significant 4 -f $gold_amr $oracle_folder/train_oracle.amr
