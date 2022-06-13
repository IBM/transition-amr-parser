# Set variables and environment for a give experiment
#
# Variables intended to be use outside of this script are CAPITALIZED. For a
# quick vim listing :g/^[A-Z]\+
#
# all paths are relative to repository root
#
set -o errexit
set -o pipefail
set -o nounset

# this will be name of the model folder
config_name=amr2joint_ontowiki2_g2g-structured-bart-large

##############################################################################
# DATA
##############################################################################

# Original AMR files in PENMAN notation
# see preprocess/README.md to create these from LDC folders
# This step will be ignored if the aligned train file below exists

# Example AMR2.0 AMR1.0 dep-parsing CFG
TASK_TAG=amr2joint_ontowiki2_g2g

# TODO: Omit these global vars and use 
# CORPUS_FOLDER=DATA/$TASK_TAG/corpora/
AMR_TRAIN_FILE_WIKI=DATA/$TASK_TAG/corpora/train.txt 
AMR_DEV_FILE_WIKI=DATA/$TASK_TAG/corpora/dev.txt 
AMR_TEST_FILE_WIKI=DATA/$TASK_TAG/corpora/test.txt

##############################################################################
# AMR ALIGNMENT
##############################################################################

# cofill: combination of JAMR and EM plus filling of missing alignments
align_tag=cofill

# All data in this step under (TODO)
ALIGNED_FOLDER=DATA/$TASK_TAG/aligned/${align_tag}/

# aligned AMR

# TODO: Omit these and use ALIGNED_FOLDER
AMR_TRAIN_FILE=$ALIGNED_FOLDER/train.txt
AMR_DEV_FILE=$ALIGNED_FOLDER/dev.txt 
AMR_TEST_FILE=$ALIGNED_FOLDER/test.txt

# wiki prediction files to recompose final AMR
# TODO: External cache, avoid external paths
# TODO: Omit these global vars and use ALIGNED_FOLDER
WIKI_DEV="$ALIGNED_FOLDER/dev.wiki"
WIKI_TEST="$ALIGNED_FOLDER/test.wiki"

##############################################################################
# ORACLE
##############################################################################

# Number of alignment samples used
ALIGNMENT_FLAGS=""
# Use importance weighted
IMPORTANCE_WEIGTHED_SAMPLING_FLAG=""

# oracle action sequences
ORACLE_TAG=bartsv-nodesplit_o10_act-states

# All data in this step under 
ORACLE_FOLDER=DATA/$TASK_TAG/oracles/${align_tag}_$ORACLE_TAG/

# Labeled SHIFT multi-task
# Top MAX_WORDS used for multi-task
MAX_WORDS=0
# Entities that will not be splitted
#ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"

# TODO: Explain this
USE_PRED_RULES=0

# Use COPY mechanism
USE_COPY=1

##############################################################################
# PRETRAINED EMBEDDINGS
##############################################################################

embedding_tag=${align_tag}_bart.large

# All data in this step under 
# FIXME: alig/oracle may alter text, we have to watch out for this
EMB_FOLDER=DATA/$TASK_TAG/embeddings/${embedding_tag}

# Pretrained embeddings 
PRETRAINED_EMBED=bart.large
PRETRAINED_EMBED_DIM=1024   # used ???
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12"
# pre-stored pretrained en embeddings (not changing with oracle)

##############################################################################
# EXTRACTED FEATURES
##############################################################################

# fairseq will extract all data into binary form

features_tag=${align_tag}_${ORACLE_TAG}_${embedding_tag}/

# all data in this step under
DATA_FOLDER=DATA/$TASK_TAG/features/$features_tag/

# Use this to feed modified source and target dicts so that embeddings match in
# fine-tuning
# TODO: see below, better return to all arguments given below. Simplified this and other like --fp16
FAIRSEQ_PREPROCESS_FINETUNE_ARGS=""

##############################################################################
# MODEL ARCHITECTURE
##############################################################################

# TODO: This is a model variable, right?
TASK=amr_action_pointer_bartsv

##### model configuration
shift_pointer_value=1
apply_tgt_actnode_masks=0
tgt_vocab_masks=1
share_decoder_embed=1     # share decoder input and output embeddings
share_all_embeddings=1    # share encoder and decoder input embeddings

arch=transformer_tgt_pointer_bartsv_large

initialize_with_bart=1
initialize_with_bart_enc=1
initialize_with_bart_dec=1
bart_encoder_backprop=1
bart_emb_backprop=1
# bart_emb_decoder=0
# bart_emb_decoder_input=0
bart_emb_init_composition=1

pointer_dist_decoder_selfattn_layers="11"
pointer_dist_decoder_selfattn_heads=1
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=11

apply_tgt_src_align=1
tgt_src_align_layers="0 1 2 3 4 5 6 7 8 9 10 11"
tgt_src_align_heads=2
tgt_src_align_focus="p0c1n0 p0c0n*"
# previous version: 'p0n1', 'p1n1' (alignment position, previous 1 position, next 1 position)
# current version: 'p0c1n1', 'p1c1n1', 'p*c1n0', 'p0c0n*', etc.
#                  'p' - previous (prior to alignment), a number or '*' for all previous src tokens
#                  'c' - current (alignment position, 1 for each tgt token), either 0 or 1
#                  'n' - next (post alignment), a number or '*' for all the remaining src tokens

apply_tgt_input_src=0
tgt_input_src_emb=top
tgt_input_src_backprop=1
tgt_input_src_combine="add"

SEEDS="42 43 44"
MAX_EPOCH=11
EVAL_INIT_EPOCH=1
time_max_between_epochs=30

# TODO: New
use_fp16=1
lr=0.0001
max_tokens=1024
update_freq=8
warmup=4000
dropout=0.2

# NEW from train 
# for apt-bart shared vocabulary
node_freq_min=5
src_roberta_emb=0
tgt_factored_emb_out=0
bart_emb_composition_pred=0
src_pool_wp2w=top
src_avg_layers=""
src_roberta_enc=0
src_fix_emb_use=0
clip_norm=0.0
weight_decay=0.0
loss_coef=1
dyo_run_start=0
dyo_run_freq=1

# FINE-TUNE ARGUMENTS
# Use this to load a pre-trained model
# TODO: see below, better return to all arguments given below. Simplified this and other like --fp16
FAIRSEQ_TRAIN_FINETUNE_ARGS=""

# AUTO NAMING <-- Avoidable?
##### set the experiment dir name based on model configurations

if [[ $pointer_dist_decoder_selfattn_layers == "0 1 2 3 4 5 6 7 8 9 10 11" ]]; then
    lay="all"
else
    lay=""
    for n in $pointer_dist_decoder_selfattn_layers; do
        [[ $n < 0 || $n > 11 ]] && echo "Invalid 'pointer_dist_decoder_selfattn_layers' input: $pointer_dist_decoder_selfattn_layers" && exit 1
        lay=$lay$(( $n + 1 ))
    done
fi


if [[ $tgt_src_align_layers == "0 1 2 3 4 5 6 7 8 9 10 11" ]]; then
    cam_lay="all"
else
    cam_lay=""
    for n in $tgt_src_align_layers; do
        [[ $n < 0 || $n > 11 ]] && echo "Invalid 'tgt_src_align_layers' input: $tgt_src_align_layers" && exit 1
        cam_lay=$cam_lay$(( $n + 1 ))
    done
fi


if [[ $tgt_src_align_focus == "p0c1n0" ]]; then
    cam_focus=""    # default
elif [[ $tgt_src_align_focus == "p0c1n0 p0c0n*" ]]; then
    cam_focus=-abuf    # alignment and "buffer"
fi

# set the experiment directory name
expdir=exp_${features_tag}_act-pos_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

# pointer distribution
ptr_tag=_ptr-lay${lay}-h${pointer_dist_decoder_selfattn_heads}    # action-pointer

if [[ $pointer_dist_decoder_selfattn_avg == 1 ]]; then
    ptr_tag=${ptr_tag}-avg
elif [[ $pointer_dist_decoder_selfattn_avg == "-1" ]]; then
    ptr_tag=${ptr_tag}-apd
fi

if [[ $apply_tgt_actnode_masks == 1 ]]; then
    ptr_tag=${ptr_tag}-pmask1
fi

# cross-attention alignment
if [[ $apply_tgt_src_align == 1 ]]; then
    cam_tag=_cam-lay${cam_lay}-h${tgt_src_align_heads}${cam_focus}
else
    cam_tag=""
fi

# target input augmentation
if [[ $apply_tgt_input_src == 1 ]]; then
    tis_tag=_tis-emb${tgt_input_src_emb}-com${tgt_input_src_combine}-bp${tgt_input_src_backprop}
else
    tis_tag=""
fi

# initialize with bart
if [[ $initialize_with_bart == 0 ]]; then
    init_tag=_bart-init${initialize_with_bart}
else
    if [[ $initialize_with_bart_enc == 0 ]]; then
        [[ $initialize_with_bart_dec == 0 ]] && echo "initialize_with_bart_dec should be 1 here" && exit 1
        init_tag=_bart-init-enc0
    fi
    if [[ $initialize_with_bart_dec == 0 ]]; then
        [[ $initialize_with_bart_enc == 0 ]] && echo "initialize_with_bart_enc should be 1 here" && exit 1
        init_tag=_bart-init-dec0
    fi
    if [[ $initialize_with_bart_enc == 1 ]] && [[ $initialize_with_bart_dec == 1 ]]; then
        init_tag=""
    fi
fi

# fix bart encoder
if [[ $bart_encoder_backprop == 0 ]]; then
    [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
    enc_fix_tag=_bart-enc-fix
else
    enc_fix_tag=""
fi

# fix bart embedding
if [[ $bart_emb_backprop == 0 ]]; then
    [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
    emb_fix_tag=_bart-emb-fix
else
    emb_fix_tag=""
fi

# decoder input and output embedding tie (encoder and decoder embeddings are always tied)
if [[ $share_decoder_embed == 0 ]]; then
    dec_emb_tag=_dec-emb-io-sep
else
    dec_emb_tag=""
fi


# initialize target embedding with compositional sub-token embeddings
if [[ $bart_emb_init_composition == 1 ]]; then
    dec_emb_init_tag="_bart-init-addi-emb"
else
    dec_emb_init_tag=""
fi

# combine different model configuration tags to the name
fp16_tag=""
if [[ $use_fp16 == 1 ]]; then
    fp16_tag="fp16-"
fi
model_tag=${expdir}${ptr_tag}${cam_tag}${tis_tag}${dec_emb_tag}${dec_emb_init_tag}${init_tag}${enc_fix_tag}${emb_fix_tag}
optim_tag=_${fp16_tag}_lr${lr}-mt${max_tokens}x${update_freq}-wm${warmup}-dp${dropout}

# All data in this step under
MODEL_FOLDER=DATA/$TASK_TAG/models/${config_name}/

###############################################################
# ENTITY LINKING
###############################################################

# Smatch evaluation with wiki

# Old scorer
LINKER_CACHE_PATH=DATA/EL/legacy_linker_amr2.0/

# BLINK
# LINKER_CACHE_PATH=DATA/EL/BLINK/linkcache

###############################################################
# TESTS 
###############################################################

##### decoding configuration for the final model
BATCH_SIZE=128
BEAM_SIZE=10
# Smatch evaluation with wiki
EVAL_METRIC=wiki.smatch
DECODING_CHECKPOINT=checkpoint_${EVAL_METRIC}_top5-avg.pt
