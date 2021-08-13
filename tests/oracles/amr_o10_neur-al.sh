#
# Unit test for the oracle with neural aligners
#

set -o errexit 
set -o pipefail
if [ -z $1 ];then 

    # Standard mini-test with wiki25, sampling
    config=configs/wiki25-structured-bart-base-neur-al-sampling.sh

else

    # custom config mini-test
    config=$1
fi
. set_environment.sh
set -o nounset

# load config
. $config

# Reuiqres a trained model
[ ! -f "$ALIGNED_FOLDER/model.pt" ] \
    && echo -e "\nRun bash test/neural_aligner.sh $config\n" \
    && exit 1

if [ ! -f "$ALIGNED_FOLDER/alignment.trn.pretty" ];then

    # Align
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGN_MODEL \
        --load-flags $ALIGN_MODEL_FLAGS \
        --vocab-text $ALIGN_VOCAB_TEXT \
        --vocab-amr $ALIGN_VOCAB_AMR \
        --write-single \
        --single-input ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --single-output $AMR_TRAIN_FILE
    
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

#    --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
#    --no-jamr \

python transition_amr_parser/amr_machine.py \
    --in-aligned-amr $AMR_TRAIN_FILE \
    --in-alignment-probs $ALIGNED_FOLDER/alignment.trn.pretty \
    --alignment-sampling-temp 1.0 \
    --out-machine-config $ORACLE_FOLDER/machine_config.json \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-tokens $ORACLE_FOLDER/train.tokens \
    --absolute-stack-positions  \
    --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
    --use-copy ${USE_COPY} \
    --reduce-nodes all

python transition_amr_parser/amr_machine.py \
    --in-machine-config $ORACLE_FOLDER/machine_config.json \
    --in-tokens $ORACLE_FOLDER/train.tokens \
    --in-actions $ORACLE_FOLDER/train.actions \
    --out-amr $ORACLE_FOLDER/train_oracle.amr

# Score
echo "Computing Smatch (may take long for 1K or more sentences)"
smatch.py -r 10 --significant 4 -f ${AMR_TRAIN_FILE_WIKI}.no_wiki $ORACLE_FOLDER/train_oracle.amr
