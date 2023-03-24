set -o errexit
set -o pipefail
. set_environment.sh
[ -z $1 ] && echo "$0 <amr_file (no wiki)>" && exit 1
gold_amr=$1
set -o nounset 

oracle_folder=DATA/AMR3.0/oracles/o10_pinitos_doc_v0.4/
mkdir -p $oracle_folder 
NORM=no-merge
#make_doc_amr
python scripts/doc-amr/get_doc_amr_from_sen.py \
            --in-amr $gold_amr \
            --coref-fof DATA/AMR3.0/coref/train_coref.fof \
            --fof-path DATA/AMR3.0/amr_annotation_3.0/ \
            --norm $NORM \
            --out-amr $oracle_folder/train_${NORM}.docamr

# get actions from oracle
python src/transition_amr_parser/amr_machine.py \
    --in-aligned-amr $oracle_folder/train_${NORM}.docamr \
    --out-machine-config $oracle_folder/machine_config.json \
    --out-actions $oracle_folder/train.actions \
    --out-tokens $oracle_folder/train.tokens \
    --use-copy 1 \
    --absolute-stack-positions
    # --reduce-nodes all

# play actions on state machine
python src/transition_amr_parser/amr_machine.py \
    --in-machine-config $oracle_folder/machine_config.json \
    --in-tokens $oracle_folder/train.tokens \
    --in-actions $oracle_folder/train.actions \
    --out-amr $oracle_folder/train_oracle_no-merge.amr

sed 's@\~[0-9]\{1,\}@@g' $oracle_folder/train_oracle_no-merge.amr > $oracle_folder/train_oracle_no-merge.amr.no_isi
sed 's@\~[0-9]\{1,\}@@g' $oracle_folder/train_no-merge.docamr > $oracle_folder/train_no-merge.docamr.no_isi
# score
echo "Computing Smatch (make take long for 1K or more sentences)"
doc-smatch -r 1 --significant 4 --coref-subscore \
                -f $oracle_folder/train_${NORM}.docamr.no_isi \
                $oracle_folder/train_oracle_no-merge.amr.no_isi \

printf "[\033[92m OK \033[0m] $0\n"
