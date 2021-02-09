# Set variables and environment for a given experiment
#
# Variables intended to be use outside of this script are CAPITALIZED
#
set -o errexit
set -o pipefail
set -o nounset

# Pretrained model we start from
# Note that copies of these files will be created in the model folder and
# modified to macoth vocabulary and embeddings
pretrain_tag=amr2.0
base_model=DATA/AMR/models/amr2.0-cofill_o5_RoBERTa-large-top24_stnp6x6-seed42/
PRETRAINED_MODEL=$base_model/checkpoint_top3-average.pt
PRETRAINED_SOURCE_DICT=$base_model/dict.en.txt
PRETRAINED_TARGET_DICT=$base_model/dict.actions.txt
# see RESTORE_FILE, SRC_DICT, TGT_DICT below for the local paths used 

TASK_TAG=AMR

# All data stored here
data_root=DATA/$TASK_TAG/

# Original AMR files in PENMAN notation
# see preprocess/README.md to create these from LDC folders
# This step will be ignored if the AMR_TRAIN_FILE file below exists
corpus_tag=bio_amr2.0
corpus_folder=$data_root/corpora/$corpus_tag/
AMR_TRAIN_FILE_WIKI=$corpus_folder/train.txt 
AMR_DEV_FILE_WIKI=$corpus_folder/dev.txt 
AMR_TEST_FILE_WIKI=$corpus_folder/test.txt

# AMR files without wiki and aligned. This will be the ones fed to the oracle
# JAMR alignments plus Pourdamghani's EM aligner plus force alignment of
# unaligned nodes
align_tag=cofill
AMR_TRAIN_FILE=$corpus_folder/train.no_wiki.aligned_${align_tag}.txt
AMR_DEV_FILE=$corpus_folder/dev.no_wiki.aligned_${align_tag}.txt 
AMR_TEST_FILE=$corpus_folder/test.no_wiki.aligned_${align_tag}.txt
# wiki cache to recompose final AMR
BLINK_CACHE_PATH=DATA/EL/BLINK/linkcache

# Labeled shift: each time we shift, we also predict the word being shited
# but restrict this to top MAX_WORDS. Controlled by
# --multitask-max-words --out-multitask-words --in-multitask-words
# To have an action calling external lemmatizer (SpaCy)
# --copy-lemma-action
ORACLE_TAG=${corpus_tag}-${align_tag}_o5
ORACLE_FOLDER=$data_root/oracles/${ORACLE_TAG}/
ORACLE_TRAIN_ARGS="
    --copy-lemma-action
"
ORACLE_DEV_ARGS="
    --copy-lemma-action
"
# If this file does not exist, it will be created from the corpus on this
# location
ENTITY_RULES="$ORACLE_FOLDER/entity_rules.json"

# PREPROCESSING
# See fairseq/fairseq/options.py:add_preprocess_args
PREPRO_TAG="RoBERTa-large-top24-finetune${pretrain_tag}"
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
FEATURES_FOLDER=$data_root/features/${ORACLE_TAG}_${PREPRO_TAG}/
# Where the pretrained dicts and models will be stored here
# (we need local copies because they will be modified)
SRC_DICT=${FEATURES_FOLDER}/dict.en.${pretrain_tag}+${ORACLE_TAG}.txt
TGT_DICT=${FEATURES_FOLDER}/dict.actions.${pretrain_tag}+${ORACLE_TAG}.txt
#
FAIRSEQ_PREPROCESS_ARGS="
    --source-lang en
    --target-lang actions
    --trainpref $ORACLE_FOLDER/train
    --validpref $ORACLE_FOLDER/dev
    --testpref $ORACLE_FOLDER/test
    --destdir $FEATURES_FOLDER
    --workers 1
    --pretrained-embed roberta.large
    --bert-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    --machine-type AMR 
    --machine-rules $ORACLE_FOLDER/train.rules.json 
    --entity-rules $ENTITY_RULES
    --srcdict $SRC_DICT
    --tgtdict $TGT_DICT
"

# TRAINING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# model types defined in ./fairseq/fairseq/models/transformer.py
TRAIN_TAG=stnp6x6-finetune
base_model=stack_transformer_6x6_nopos
# number of random seeds trained at once
NUM_SEEDS=3
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
# --lazy-load for very large corpora (data does not fit into RAM)
# --bert-backprop do backprop though BERT
# NOTE: --save-dir is specified inside dcc/train.sh to account for the seed
MAX_EPOCH=170
CHECKPOINTS_DIR_ROOT="$data_root/models/${ORACLE_TAG}_${PREPRO_TAG}_${TRAIN_TAG}"
# Note pretrained checkpoint is common to all seeds, its stored in the features folder
RESTORE_FILE=${FEATURES_FOLDER}/checkpoint_${pretrain_tag}+${ORACLE_TAG}.pt
FAIRSEQ_TRAIN_ARGS="
    $FEATURES_FOLDER
    --restore-file $RESTORE_FILE
    --reset-optimizer
    --max-epoch $MAX_EPOCH
    --arch $base_model
    --optimizer adam
    --adam-betas '(0.9,0.98)'
    --clip-norm 0.0
    --lr-scheduler inverse_sqrt
    --warmup-init-lr 1e-07
    --warmup-updates 4000
    --pretrained-embed-dim 1024
    --lr 0.0005
    --min-lr 1e-09
    --dropout 0.3
    --weight-decay 0.0
    --criterion label_smoothed_cross_entropy
    --label-smoothing 0.01
    --keep-last-epochs 40
    --max-tokens 3584
    --log-format json
    --fp16
"

# TESTING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# --path flag specified in the dcc/test.sh script
# --results-path is dirname from --path plus $TEST_TAG
beam_size=1
TEST_TAG="beam${beam_size}"
CHECKPOINT=checkpoint_best_WIKI.SMATCH.pt
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
FAIRSEQ_GENERATE_ARGS="
    $FEATURES_FOLDER 
    --gen-subset valid
    --machine-type AMR 
    --machine-rules $ORACLE_FOLDER/train.rules.json
    --entity-rules $ENTITY_RULES
    --beam ${beam_size}
    --batch-size 128
    --remove-bpe
"
