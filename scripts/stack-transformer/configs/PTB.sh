# Set variables and environment for a give experiment
set -o errexit
set -o pipefail
set -o nounset

# Oracles are precomputed ans stored here
PTB_ORACLE=/dccstor/ykt-parse/SHARED/MODELS/dep-parsing/transition-amr-parser/oracles/

TASK_TAG=dep-parsing

# All data stored here
data_root=DATA/$TASK_TAG/

# Dependency-parsing oracle
# NOTE: This is precomputed
# ORACLE_TAG=PTB_SD_3_3_0
ORACLE_TAG=PTB_SD_3_3_0+Word100
ORACLE_FOLDER=$data_root/oracles/${ORACLE_TAG}/

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
    --tokenize-by-whitespace
    --machine-type $TASK_TAG
"

# TRAINING
# See fairseq/fairseq/options.py:add_optimization_args,add_checkpoint_args
# model types defined in ./fairseq/fairseq/models/transformer.py
TRAIN_TAG=stops6x6
base_model=stack_transformer_6x6_tops_nopos
# number of random seeds trained at once
NUM_SEEDS=3
# CCC configuration in scripts/stack-transformer/jbsub_experiment.sh
TRAIN_GPU_TYPE=v100
TRAIN_QUEUE=ppc_24h
# --lazy-load for very large corpora (data does not fit into RAM)
# --bert-backprop do backprop though BERT
# NOTE: --save-dir is specified inside dcc/train.sh to account for the seed
MAX_EPOCH=50
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
    --keep-last-epochs 30
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
    --machine-type $TASK_TAG
    --beam ${beam_size}
    --batch-size 128
    --remove-bpe
"
# TODO: It would be cleaner to use the checkpoint path for --machine-rules but
# this can be externally provided on dcc/test.sh
