set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# prepare data
rm -R DATA.tmp/neural_aligner/
mkdir -p DATA.tmp/neural_aligner/
FOLDER=DATA.tmp/neural_aligner/
cp DATA/wiki25.jkaln $FOLDER/wiki25.amr

# Preprocess
# Extract ELMO vocabulary
python align_cfg/vocab.py --in-amrs $FOLDER/wiki25.amr --out-folder $FOLDER
# Extract ELMO embeddings
bash align_cfg/pretrained_embeddings.sh $FOLDER 

# TODO: learn alignments

# align data
python -u align_cfg/main.py \
    --cuda \
    --cache-dir $FOLDER \
	--log-dir $FOLDER/version_20210709c_exp_0_seed_0_write_amr2  \
	--model-config '{"text_emb": "char", "text_enc": "bilstm", "text_project": 200, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 200, "dropout": 0.3, "context": "xy", "hidden_size": 200, "prior": "attn", "output_mode": "tied"}' \
	--batch-size 8 \
	--accum-steps 16 \
	--lr 0.0001 \
	--max-length 100 \
	--verbose \
	--max-epoch 200 \
	--pr 0 \
	--pr-after 1000 \
	--pr-mode posterior \
	--seed 53060822 \
	--name version_20210709c_exp_0_seed_0 \
	--load /dccstor/ykt-parse/SHARED/misc/adrozdov/log/align/version_20210709c_exp_0_seed_0/model.best.val_1_recall.pt \
	--trn-amr $FOLDER/wiki25.amr \
	--write-only \
	--batch-size 8 \
	--max-length 0
