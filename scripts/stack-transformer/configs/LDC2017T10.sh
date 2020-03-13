# Set variables and environment for a give experiment
#
# Variables intended to be use outside of this script are CAPITALIZED
#
set -o errexit
set -o pipefail
set -o nounset

TASK_TAG=AMR

# All data stored here
data_root=DATA/$TASK_TAG/

LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/

# AMR ORACLE
# See transition_amr_parser/data_oracle.py:argument_parser
# NOTE: LDC2016_AMR_CORPUS should be defined in set_envinroment.sh
AMR_TRAIN_FILE=$LDC2016_AMR_CORPUS/jkaln_2016_scr.txt 
AMR_DEV_FILE=$LDC2016_AMR_CORPUS/dev.txt.removedWiki.noempty.JAMRaligned 
AMR_TEST_FILE=$LDC2016_AMR_CORPUS/test.txt.removedWiki.noempty.JAMRaligned
# WIKI files
WIKI_DEV=/dccstor/multi-parse/amr/dev.wiki
AMR_DEV_FILE_WIKI=/dccstor/ykt-parse/AMR/2016data/dev.txt 
WIKI_TEST=/dccstor/multi-parse/amr/test.wiki
AMR_TEST_FILE_WIKI=/dccstor/ykt-parse/AMR/2016data/test.txt

# Labeled shift: each time we shift, we also predict the word being shited
# but restrict this to top MAX_WORDS. Controlled by
# --multitask-max-words --out-multitask-words --in-multitask-words
# To have an action calling external lemmatizer (SpaCy)
# --copy-lemma-action
MAX_WORDS=100
ORACLE_TAG=o3+Word${MAX_WORDS}
ORACLE_FOLDER=$data_root/oracles/${ORACLE_TAG}/
ORACLE_TRAIN_ARGS="
    --multitask-max-words $MAX_WORDS 
    --out-multitask-words $ORACLE_FOLDER/train.multitask_words 
    --copy-lemma-action
"
ORACLE_DEV_ARGS="
    --in-multitask-words $ORACLE_FOLDER/train.multitask_words \
    --copy-lemma-action
"

# GPU
# k80, v100 (3 times faster)

# PREPROCESSING
# See fairseq/fairseq/options.py:add_preprocess_args
PREPRO_TAG="RoBERTa-base"
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
PREPRO_GPU_TYPE=v100
PREPRO_QUEUE=x86_6h
features_folder=$data_root/features/${ORACLE_TAG}_${PREPRO_TAG}/
FAIRSEQ_PREPROCESS_ARGS="
    --source-lang en
    --target-lang actions
    --trainpref $ORACLE_FOLDER/train
    --validpref $ORACLE_FOLDER/dev
    --testpref $ORACLE_FOLDER/test
    --destdir $features_folder 
    --workers 1 
    --machine-type AMR \
    --machine-rules $ORACLE_FOLDER/train.rules.json 
"

# TRAINING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# model types defined in ./fairseq/fairseq/models/transformer.py
TRAIN_TAG=stops6x6
base_model=stack_transformer_6x6_tops_nopos
# number of random seeds trained at once
NUM_SEEDS=1
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
TRAIN_GPU_TYPE=v100
TRAIN_QUEUE=ppc_24h
# --lazy-load for very large corpora (data does not fit into RAM)
# --bert-backprop do backprop though BERT
# NOTE: --save-dir is specified inside dcc/train.sh to account for the seed
MAX_EPOCH=100
CHECKPOINTS_DIR_ROOT="$data_root/models/${ORACLE_TAG}_${PREPRO_TAG}_${TRAIN_TAG}"
FAIRSEQ_TRAIN_ARGS="
    $features_folder
    --max-epoch $MAX_EPOCH
    --arch $base_model
    --optimizer adam
    --adam-betas '(0.9,0.98)'
    --clip-norm 0.0
    --lr-scheduler inverse_sqrt
    --warmup-init-lr 1e-07
    --warmup-updates 4000
    --pretrained-embed-dim 768
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
CHECKPOINT=checkpoint_best.pt
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
TEST_GPU_TYPE=v100
TEST_QUEUE=x86_6h
FAIRSEQ_GENERATE_ARGS="
    $features_folder 
    --gen-subset valid
    --machine-type AMR 
    --machine-rules $ORACLE_FOLDER/train.rules.json
    --beam ${beam_size}
    --batch-size 128
    --remove-bpe
"
# TODO: It would be cleaner to use the checkpoint path for --machine-rules but
# this can be externally provided on dcc/test.sh
