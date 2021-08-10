#
# Unit test for the oracle with neural aligners
#

set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset 

config=configs/wiki25-neur-al-sampling.sh

# load config
. $config

# Reuiqres a trained model
[ ! -f "$ALIGNED_FOLDER/model.pt" ] \
    && echo -e "\nRun bash test/neural_aligner.sh $config\n" \
    && exit 1

if [ ! -f "$ALIGNED_FOLDER/alignment.trn.pretty" ];then
    
    # Get alignment probabilities
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGN_MODEL \
        --load-flags $ALIGN_MODEL_FLAGS \
        --vocab-text $ALIGN_VOCAB_TEXT \
        --vocab-amr $ALIGN_VOCAB_AMR \
        --trn-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --val-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --log-dir $ALIGNED_FOLDER \
        --write-pretty
    
fi

mkdir -p $ORACLE_FOLDER

python transition_amr_parser/amr_machine.py \
    --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --amr-from-penman \
    --in-alignment-probs $ALIGNED_FOLDER/alignment.trn.pretty \
    --out-machine-config $ORACLE_FOLDER/machine_config.json \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-tokens $ORACLE_FOLDER/train.tokens \
    --absolute-stack-positions  \
    # --reduce-nodes all

python transition_amr_parser/amr_machine.py \
    --in-machine-config $ORACLE_FOLDER/machine_config.json \
    --in-tokens $ORACLE_FOLDER/train.tokens \
    --in-actions $ORACLE_FOLDER/train.actions \
    --out-amr $ORACLE_FOLDER/train_oracle.amr

# Score
echo "Conmputing Smatch (make take long for 1K or more sentences)"
smatch.py -r 10 --significant 4 -f ${AMR_TRAIN_FILE_WIKI}.no_wiki $ORACLE_FOLDER/train_oracle.amr
