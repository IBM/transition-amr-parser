#
# Unit test for the oracle with neural aligners
#

set -o errexit 
set -o pipefail
if [ -z $1 ];then 

    # Standard mini-test with wiki25, sampling
    config=configs/amr2.0-structured-bart-large-counts-al.sh

else

    # custom config mini-test
    config=$1
fi
. set_environment.sh
set -o nounset

# load config
. $config

TEMPERATURE=0.0

# remove wiki, tokenize sentences unless we use JAMR reference
# train
python preprocess/remove_wiki.py \
    $AMR_TRAIN_FILE_WIKI \
    ${AMR_TRAIN_FILE_WIKI}.no_wiki
python scripts/tokenize_amr.py --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki
# dev 
python preprocess/remove_wiki.py \
    $AMR_DEV_FILE_WIKI \
    ${AMR_DEV_FILE_WIKI}.no_wiki
python scripts/tokenize_amr.py --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki
# test
python preprocess/remove_wiki.py \
    $AMR_TEST_FILE_WIKI \
    ${AMR_TEST_FILE_WIKI}.no_wiki
python scripts/tokenize_amr.py --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki


# Train model if missing
if [ ! -f "$ALIGNED_FOLDER/model.json" ];then

    # train aligner
    python transition_amr_parser/amr_aligner.py \
        --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --out-checkpoint-json $ALIGNED_FOLDER/model.json
    
fi

# Align data
# train
python transition_amr_parser/amr_aligner.py \
    --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --in-checkpoint-json $ALIGNED_FOLDER/model.json \
    --em-epochs 0 \
    --out-aligned-amr $AMR_TRAIN_FILE \
    --out-alignment-probs $ALIGNED_FOLDER/alignment.trn.pretty 
# dev
python transition_amr_parser/amr_aligner.py \
    --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki \
    --in-checkpoint-json $ALIGNED_FOLDER/model.json \
    --em-epochs 0 \
    --out-aligned-amr $AMR_DEV_FILE 
# test
python transition_amr_parser/amr_aligner.py \
    --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki \
    --in-checkpoint-json $ALIGNED_FOLDER/model.json \
    --em-epochs 0 \
    --out-aligned-amr $AMR_TEST_FILE 
    
mkdir -p $ORACLE_FOLDER

python transition_amr_parser/amr_machine.py \
    --in-aligned-amr $AMR_TRAIN_FILE \
    --in-alignment-probs $ALIGNED_FOLDER/alignment.trn.pretty \
    --alignment-sampling-temp $TEMPERATURE \
    --out-machine-config $ORACLE_FOLDER/machine_config.json \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-tokens $ORACLE_FOLDER/train.tokens \
    --absolute-stack-positions  \
    --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
    --use-copy ${USE_COPY} \
    # --reduce-nodes all

python transition_amr_parser/amr_machine.py \
    --in-machine-config $ORACLE_FOLDER/machine_config.json \
    --in-tokens $ORACLE_FOLDER/train.tokens \
    --in-actions $ORACLE_FOLDER/train.actions \
    --out-amr $ORACLE_FOLDER/train_oracle.amr

# Score
echo "Computing Smatch (may take long for 1K or more sentences)"
smatch.py -r 10 --significant 4 -f ${AMR_TRAIN_FILE_WIKI}.no_wiki $ORACLE_FOLDER/train_oracle.amr
