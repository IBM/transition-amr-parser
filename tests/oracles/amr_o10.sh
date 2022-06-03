set -o errexit
set -o pipefail
. set_environment.sh
[ -z $1 ] && echo "$0 <amr_file (no wiki)>" && exit 1
gold_amr=$1
set -o nounset 

oracle_folder=DATA/unit_test_$(basename $(dirname $gold_amr))/
mkdir -p $oracle_folder 
 
# get actions from oracle
python transition_amr_parser/amr_machine.py \
    --in-aligned-amr $gold_amr \
    --out-machine-config $oracle_folder/machine_config.json \
    --out-actions $oracle_folder/train.actions \
    --out-tokens $oracle_folder/train.tokens \
    --use-copy 1 \
    --absolute-stack-positions  \
    # --if-oracle-error stop
    # --reduce-nodes all

# play actions on state machine
python transition_amr_parser/amr_machine.py \
    --in-machine-config $oracle_folder/machine_config.json \
    --in-tokens $oracle_folder/train.tokens \
    --in-actions $oracle_folder/train.actions \
    --out-amr $oracle_folder/train_oracle.amr

# score
echo "Computing Smatch (make take long for 1K or more sentences)"
python scripts/smatch_aligner.py \
    --in-amr $oracle_folder/train_oracle.amr \
    --in-reference-amr $gold_amr \
    # --stop-if-different

printf "[\033[92m OK \033[0m] $0\n"
