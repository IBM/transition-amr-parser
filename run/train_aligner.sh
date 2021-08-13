#!/bin/bash
set -o errexit
set -o pipefail

# Argument handling
HELP="\nbash $0 <config>\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
config=$1
[ ! -f "$config" ] && "Missing $config" && exit 1

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset

# Load config
echo "[Configuration file:]"
echo $config
. $config 

# We will need this to save the alignment log
[ ! -d "$ALIGNED_FOLDER" ] && mkdir -p $ALIGNED_FOLDER

# remove wiki, tokenize sentences unless we use JAMR reference
python preprocess/remove_wiki.py $AMR_TRAIN_FILE_WIKI ${AMR_TRAIN_FILE_WIKI}.no_wiki
python preprocess/remove_wiki.py $AMR_DEV_FILE_WIKI ${AMR_DEV_FILE_WIKI}.no_wiki
python preprocess/remove_wiki.py $AMR_TEST_FILE_WIKI ${AMR_TEST_FILE_WIKI}.no_wiki
# tokenize
# TODO:  This assumes we provide JAMR tokenization (for now)
#if [ false ];then
python scripts/tokenize_amr.py --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki
python scripts/tokenize_amr.py --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki
python scripts/tokenize_amr.py --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki
#fi

# generate ELMO vocabulary
python align_cfg/vocab.py \
    --in-amrs ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --out-text $ALIGN_VOCAB_TEXT \
    --out-amr $ALIGN_VOCAB_AMR

# Generate embeddings for the aligner
python align_cfg/pretrained_embeddings.py --cuda \
    --cache-dir $ALIGNED_FOLDER \
    --vocab $ALIGN_VOCAB_TEXT
python align_cfg/pretrained_embeddings.py --cuda \
    --cache-dir $ALIGNED_FOLDER \
    --vocab $ALIGN_VOCAB_AMR

# Learn alignments.
python -u align_cfg/main.py --aligner-training-and-eval \
    --cuda --allow-cpu \
    --vocab-text $ALIGN_VOCAB_TEXT \
    --vocab-amr $ALIGN_VOCAB_AMR \
    --trn-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --val-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
    --cache-dir $ALIGNED_FOLDER \
    --log-dir $ALIGNED_FOLDER/test_train_aligner \
    --model-config '{"text_emb": "char", "text_enc": "bilstm", "text_project": 20, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 20, "dropout": 0.3, "context": "xy", "hidden_size": 20, "prior": "attn", "output_mode": "tied"}' \
    --batch-size 4 \
    --accum-steps 2 \
    --lr 0.0001 \
    --max-length 100 \
    --verbose \
    --max-epoch 5 \
    --seed 12345

# Copy learned model and config
cp $ALIGNED_FOLDER/test_train_aligner/model.best.val_0_recall.pt $ALIGNED_FOLDER/model.pt
cp $ALIGNED_FOLDER/test_train_aligner/flags.json $ALIGNED_FOLDER/flags.json
