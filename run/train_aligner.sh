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

# quick exit
if [ -f $ALIGNED_FOLDER/.done ]; then

    printf "[\033[92m done \033[0m] $ALIGNED_FOLDER/.done\n"
    exit 0

fi

###### AMR Alignment (Pre-processing)
if [ -f $ALIGNED_FOLDER/.done.preprocess ]; then

    printf "[\033[92m done \033[0m] $ALIGNED_FOLDER/.done.preprocess\n"

else

    # when ran isolated we need to create this ourselves
    echo "mkdir -p $ALIGNED_FOLDER"
    mkdir -p $ALIGNED_FOLDER

    # remove wiki
    python scripts/remove_wiki.py \
        $AMR_TRAIN_FILE_WIKI \
        ${AMR_TRAIN_FILE_WIKI}.no_wiki
    python scripts/remove_wiki.py \
        $AMR_DEV_FILE_WIKI \
        ${AMR_DEV_FILE_WIKI}.no_wiki
    python scripts/remove_wiki.py \
        $AMR_TEST_FILE_WIKI \
        ${AMR_TEST_FILE_WIKI}.no_wiki

    # tokenize
    python ibm_neural_aligner/tokenize_amr.py \
        --in-amr ${AMR_TRAIN_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/train.unaligned.txt
    python ibm_neural_aligner/tokenize_amr.py \
        --in-amr ${AMR_DEV_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/dev.unaligned.txt
    python ibm_neural_aligner/tokenize_amr.py \
        --in-amr ${AMR_TEST_FILE_WIKI}.no_wiki \
        --out-amr $ALIGNED_FOLDER/test.unaligned.txt

    # dummy align
    python ibm_neural_aligner/dummy_align.py \
        --in-amr $ALIGNED_FOLDER/train.unaligned.txt \
        --out-amr $ALIGNED_FOLDER/train.dummy_align.txt
    python ibm_neural_aligner/dummy_align.py \
        --in-amr $ALIGNED_FOLDER/dev.unaligned.txt \
        --out-amr $ALIGNED_FOLDER/dev.dummy_align.txt
    python ibm_neural_aligner/dummy_align.py \
        --in-amr $ALIGNED_FOLDER/test.unaligned.txt \
        --out-amr $ALIGNED_FOLDER/test.dummy_align.txt

    cp $ALIGNED_FOLDER/dev.dummy_align.txt $ALIGNED_FOLDER/dev.txt
    cp $ALIGNED_FOLDER/test.dummy_align.txt $ALIGNED_FOLDER/test.txt

    touch $ALIGNED_FOLDER/.done.preprocess

fi

###### AMR Alignment (Cache data)
if [ -f $ALIGNED_FOLDER/.done.cache_data ]; then

    printf "[\033[92m done \033[0m] $ALIGNED_FOLDER/.done.cache_data\n"

else

    # Generate text and AMR vocabulary.
    python ibm_neural_aligner/vocab.py \
        --in-amrs \
            $ALIGNED_FOLDER/train.unaligned.txt \
            $ALIGNED_FOLDER/dev.unaligned.txt \
            $ALIGNED_FOLDER/test.unaligned.txt \
        --out-text $ALIGN_VOCAB_TEXT \
        --out-amr $ALIGN_VOCAB_AMR

    # Pre-compute embeddings for text and AMR.
    python ibm_neural_aligner/pretrained_embeddings.py --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab $ALIGNED_FOLDER/vocab.text.txt
    python ibm_neural_aligner/pretrained_embeddings.py --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab $ALIGNED_FOLDER/vocab.amr.txt

    touch $ALIGNED_FOLDER/.done.cache_data

fi

###### AMR Alignment (Train aligner)
if [ -f $ALIGNED_FOLDER/.done.train ]; then

    printf "[\033[92m done \033[0m] $ALIGNED_FOLDER/.done.train\n"

else

    python ibm_neural_aligner/main.py \
        --cuda \
        --cache-dir $ALIGNED_FOLDER \
        --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
        --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
        --trn-amr $ALIGNED_FOLDER/train.dummy_align.txt \
        --val-amr $ALIGNED_FOLDER/dev.dummy_align.txt \
        --tst-amr $ALIGNED_FOLDER/test.dummy_align.txt \
        --lr 1e-4 \
        --max-length 100 \
        --log-dir $ALIGNED_FOLDER/log \
        --max-epoch 200 \
        --model-config '{"text_emb": "char", "text_enc": "bilstm", "text_project": 200, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 200, "dropout": 0.1, "context": "xy", "hidden_size": 200, "prior": "attn", "output_mode": "tied"}' \
        --batch-size 32 \
        --accum-steps 4 \
        --verbose \
        #--skip-validation

    touch $ALIGNED_FOLDER/.done.train
fi

###### AMR Alignment (Extract)
if [ -f $ALIGNED_FOLDER/.done ]; then

    printf "[\033[92m done \033[0m] $ALIGNED_FOLDER/.done\n"

else

   # Get alignment probabilities.
   python ibm_neural_aligner/main.py --cuda \
       --cache-dir $ALIGNED_FOLDER \
       --load $ALIGNED_FOLDER/log/model.latest.pt \
       --load-flags $ALIGNED_FOLDER/log/flags.json \
       --vocab-text $ALIGNED_FOLDER/vocab.text.txt \
       --vocab-amr $ALIGNED_FOLDER/vocab.amr.txt \
       --trn-amr $ALIGNED_FOLDER/train.dummy_align.txt \
       --val-amr $ALIGNED_FOLDER/train.dummy_align.txt \
       --log-dir $ALIGNED_FOLDER \
       --write-align-dist \
       --single-input $ALIGNED_FOLDER/train.dummy_align.txt \
       --single-output $ALIGNED_FOLDER/alignment.trn.align_dist.npy

    python ibm_neural_aligner/align_utils.py write_argmax \
       --in-amr $ALIGNED_FOLDER/train.dummy_align.txt \
       --in-amr-align-dist $ALIGNED_FOLDER/alignment.trn.align_dist.npy \
       --out-amr-aligned $ALIGNED_FOLDER/train.txt

   touch $ALIGNED_FOLDER/.done

fi

# python ibm_neural_aligner/align_utils.py verify_corpus_id --in-amr $ALIGNED_FOLDER/train.txt --corpus-id $ALIGNED_FOLDER/alignment.trn.align_dist.npy.corpus_hash
