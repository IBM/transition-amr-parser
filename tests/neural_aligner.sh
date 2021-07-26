set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# UNCOMMENT to start test from blank slate.
# It's preferable not to delete this directory since it forces a re-download
# of model checkpoints (i.e., elmo).
# rm -Rf DATA/wiki25/*

# Train aligner
bash run/train_aligner.sh configs/wiki25-neur-al-sampling.sh 

# load config
. configs/wiki25-neur-al-sampling.sh 

# Align data.
mkdir -p $ALIGNED_FOLDER/version_20210709c_exp_0_seed_0_write_amr2
python -u align_cfg/main.py --no-jamr \
    --cuda --allow-cpu \
    --vocab-text $ALIGN_VOCAB_TEXT \
    --vocab-amr $ALIGN_VOCAB_AMR \
    --write-single \
    --single-input ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --single-output $ALIGNED_FOLDER/version_20210709c_exp_0_seed_0_write_amr2/alignment.trn.out.pred \
    --cache-dir $ALIGNED_FOLDER \
    --verbose \
    --load $ALIGN_MODEL  \
    --load-flags $ALIGN_MODEL_FLAGS \
    --batch-size 8 \
    --max-length 0

# results should be written to
if [ -f "$FOLDER/version_20210709c_exp_0_seed_0_write_amr2/alignment.trn.out.pred" ];then
    printf "\n[\033[92mOK\033[0m] $0\n\n"
else
    printf "\n[\033[91mFAILED\033[0m] $0\n\n"
fi
