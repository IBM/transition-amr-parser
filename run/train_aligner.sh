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

###### AMR Alignment (Pre-processing)
if [ -f $ALIGNED_FOLDER/.done.preprocess ]; then

    echo "AMR Alignment (Pre-processing). Already done --- do nothing."

else

    # remove wiki
    python preprocess/remove_wiki.py \
        $AMR_TRAIN_FILE_WIKI \
        ${AMR_TRAIN_FILE_WIKI}.no_wiki
    python preprocess/remove_wiki.py \
        $AMR_DEV_FILE_WIKI \
        ${AMR_DEV_FILE_WIKI}.no_wiki
    python preprocess/remove_wiki.py \
        $AMR_TEST_FILE_WIKI \
        ${AMR_TEST_FILE_WIKI}.no_wiki

    # tokenize
    python align_cfg/tokenize_amr.py \
        --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/train.unaligned.txt
    python align_cfg/tokenize_amr.py \
        --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/dev.unaligned.txt
    python align_cfg/tokenize_amr.py \
        --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/test.unaligned.txt

    touch $ALIGNED_FOLDER/.done.preprocess

fi

###### AMR Alignment (Cache data)
if [ -f $ALIGNED_FOLDER/.done.cache_data ]; then

    echo "AMR Alignment (Cache data). Already done --- do nothing."

else

    # Generate text and AMR vocabulary.
    python align_cfg/vocab.py \
        --in-amrs \
            $ALIGNED_FOLDER/train.unaligned.txt \
            $ALIGNED_FOLDER/dev.unaligned.txt \
            $ALIGNED_FOLDER/test.unaligned.txt \
        --out-text $ALIGN_VOCAB_TEXT \
        --out-amr $ALIGN_VOCAB_AMR

    # Pre-compute embeddings for text and AMR.
    python align_cfg/pretrained_embeddings.py --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab $ALIGNED_FOLDER/vocab.text.txt
    python align_cfg/pretrained_embeddings.py --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab $ALIGNED_FOLDER/vocab.amr.txt

    touch $ALIGNED_FOLDER/.done.cache_data

fi

###### AMR Alignment (Train aligner)
if [ -f $ALIGNED_FOLDER/.done.train ]; then

    echo "AMR Alignment (Train aligner). Already done --- do nothing."

else

    python align_cfg/main.py \
        --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
        --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
        --trn-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --val-amr $ALIGNED_FOLDER/dev.unaligned.txt \
        --tst-amr $ALIGNED_FOLDER/test.unaligned.txt \
        --lr 2e-3 \
        --max-length 100 \
        --log-dir $ALIGNED_FOLDER/log \
        --max-epoch 20 \
        --model-config '{"text_emb": "char", "text_enc": "bilstm", "text_project": 200, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 200, "dropout": 0.3, "context": "xy", "hidden_size": 200, "prior": "attn", "output_mode": "tied"}' \
        --batch-size 32 \
        --accum-steps 4 \
        --verbose \
        --skip-validation

    touch $ALIGNED_FOLDER/.done.train
fi

###### AMR Alignment (Extract)
if [ -f $ALIGNED_FOLDER/.done ]; then

    echo "AMR Alignment (Extract). Already done --- do nothing."

else

    # ARGMAX alignments.
    python align_cfg/main.py --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGNED_FOLDER/log/model.latest.pt \
        --load-flags $ALIGNED_FOLDER/log/flags.json \
        --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
        --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
        --write-single \
        --single-input $ALIGNED_FOLDER/train.unaligned.txt \
        --single-output $ALIGNED_FOLDER/train.txt

    # Get alignment probabilities
    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGNED_FOLDER/log/model.latest.pt \
        --load-flags $ALIGNED_FOLDER/log/flags.json \
        --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
        --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
        --trn-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --val-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --log-dir $ALIGNED_FOLDER \
        --write-pretty

    python align_cfg/main.py --cuda \
        --no-jamr \
        --cache-dir $ALIGNED_FOLDER \
        --load $ALIGNED_FOLDER/log/model.latest.pt \
        --load-flags $ALIGNED_FOLDER/log/flags.json \
        --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
        --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
        --trn-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --val-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --log-dir $ALIGNED_FOLDER \
        --write-align-dist \
        --single-output $ALIGNED_FOLDER/alignment.trn.align_dist.npy

    touch $ALIGNED_FOLDER/.done

fi

# FIXME: Unelegant
cp DATA/$TASK_TAG/aligned/cofill/dev.txt $ALIGNED_FOLDER/dev.txt
cp DATA/$TASK_TAG/aligned/cofill/test.txt $ALIGNED_FOLDER/test.txt

# Copy learned model and config
# cp $ALIGNED_FOLDER/test_train_aligner/model.best.val_0_recall.pt $ALIGNED_FOLDER/model.pt
# cp $ALIGNED_FOLDER/test_train_aligner/flags.json $ALIGNED_FOLDER/flags.json
