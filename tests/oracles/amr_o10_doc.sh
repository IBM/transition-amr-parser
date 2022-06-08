set -o errexit
set -o pipefail
. set_environment.sh
[ -z $1 ] && echo "$0 <amr_file (no wiki)>" && exit 1
gold_amr=$1
set -o nounset 

oracle_folder=DATA/AMR3.0/oracles/o10_pinitos_doc_test/
mkdir -p $oracle_folder 
 
# get actions from oracle
python transition_amr_parser/amr_machine_docamr.py \
    --in-aligned-amr $gold_amr \
    --out-machine-config $oracle_folder/machine_config.json \
    --out-actions $oracle_folder/train.actions \
    --out-tokens $oracle_folder/train.tokens \
    --use-copy 1 \
    --absolute-stack-positions \
    --coref-fof DATA/AMR3.0/coref/train_coref.fof \
    --norm no-merge \
    --fof-path <path to AMR3.0> /
    # --reduce-nodes all

# play actions on state machine
python transition_amr_parser/amr_machine_docamr.py \
    --in-machine-config $oracle_folder/machine_config.json \
    --in-tokens $oracle_folder/train.tokens \
    --in-actions $oracle_folder/train.actions \
    --out-amr $oracle_folder/train_oracle_no-merge.amr

# score
echo "Computing Smatch (make take long for 1K or more sentences)"
# python smatch/smatch_doc.py \
#    -f 

printf "[\033[92m OK \033[0m] $0\n"
