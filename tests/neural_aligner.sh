set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# UNCOMMENT to start test from blank slate.
# It's preferable not to delete this directory since it forces a re-download
# of model checkpoints (i.e., elmo).
# rm -R DATA.tmp/neural_aligner/

# prepare data
mkdir -p DATA.tmp/neural_aligner/
FOLDER=DATA.tmp/neural_aligner/
cp DATA/wiki25.jkaln $FOLDER/wiki25.amr

# Preprocess
# Extract ELMO vocabulary
python align_cfg/vocab.py --in-amrs $FOLDER/wiki25.amr --out-folder $FOLDER
# Extract ELMO embeddings
python align_cfg/pretrained_embeddings.py \
    --cuda --allow-cpu \
    --vocab-text $FOLDER/ELMO_vocab.text.txt \
    --cache-dir $FOLDER/
python align_cfg/pretrained_embeddings.py \
    --cuda --allow-cpu \
    --vocab-text $FOLDER/ELMO_vocab.amr.txt \
    --cache-dir $FOLDER/

# Learn alignments.
python -u align_cfg/main.py --aligner-training-and-eval \
    --cuda --allow-cpu \
    --vocab-text $FOLDER/ELMO_vocab.text.txt \
    --vocab-amr $FOLDER/ELMO_vocab.amr.txt \
    --trn-amr $FOLDER/wiki25.amr \
    --val-amr $FOLDER/wiki25.amr \
    --cache-dir $FOLDER \
    --log-dir $FOLDER/test_train_aligner \
    --model-config '{"text_emb": "char", "text_enc": "bilstm", "text_project": 20, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 20, "dropout": 0.3, "context": "xy", "hidden_size": 20, "prior": "attn", "output_mode": "tied"}' \
    --batch-size 4 \
    --accum-steps 2 \
    --lr 0.0001 \
    --max-length 100 \
    --verbose \
    --max-epoch 5 \
    --seed 12345

# Align data.
mkdir -p $FOLDER/version_20210709c_exp_0_seed_0_write_amr2
python -u align_cfg/main.py --no-jamr \
    --cuda --allow-cpu \
    --vocab-text $FOLDER/ELMO_vocab.text.txt \
    --vocab-amr $FOLDER/ELMO_vocab.amr.txt \
    --write-single \
    --single-input $FOLDER/wiki25.amr \
    --single-output $FOLDER/version_20210709c_exp_0_seed_0_write_amr2/alignment.trn.out.pred \
    --cache-dir $FOLDER \
    --verbose \
    --load $FOLDER/test_train_aligner/model.best.val_0_recall.pt  \
    --load-flags $FOLDER/test_train_aligner/flags.json \
    --batch-size 8 \
    --max-length 0

# results should be written to
# DATA.tmp/neural_aligner/version_20210709c_exp_0_seed_0_write_amr2/alignment.trn.out.pred
python -c "import os; assert os.path.exists('DATA.tmp/neural_aligner/version_20210709c_exp_0_seed_0_write_amr2/alignment.trn.out.pred')"

