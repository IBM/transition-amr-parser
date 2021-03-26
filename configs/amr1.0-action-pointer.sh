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

##############################################################################
# DATA
##############################################################################

# Original AMR files in PENMAN notation
# see preprocess/README.md to create these from LDC folders
# This step will be ignored if the aligned train file below exists

# Example AMR2.0 AMR1.0 dep-parsing CFG
TASK_TAG=AMR1.0

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
WIKI_DEV=""
WIKI_TEST=""

##############################################################################
# ORACLE
##############################################################################

# oracle action sequences
ORACLE_TAG=o8.3_act-states

# All data in this step under 
ORACLE_FOLDER=DATA/$TASK_TAG/oracles/${align_tag}_$ORACLE_TAG/

# Labeled SHIFT multi-task
# Top MAX_WORDS used for multi-task
MAX_WORDS=0
# Entities that will not be splitted
ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"

# TODO: Explain this
USE_PRED_RULES=0

##############################################################################
# PRETRAINED EMBEDDINGS
##############################################################################

embedding_tag=RoBERTa-large-top24

# All data in this step under 
# FIXME: alig/oracle may alter text, we have to watch out for this
EMB_FOLDER=DATA/$TASK_TAG/embeddings/${embedding_tag}

# Pretrained embeddings 
PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
# pre-stored pretrained en embeddings (not changing with oracle)

##############################################################################
# EXTRACTED FEATURES
##############################################################################

# fairseq will extract all data into binary form

features_tag=${align_tag}_${ORACLE_TAG}_${embedding_tag}/

# all data in this step under
DATA_FOLDER=DATA/$TASK_TAG/features/$features_tag/

##############################################################################
# MODEL ARCHITECTURE
##############################################################################

# TODO: This is a model variable, right?
TASK=amr_action_pointer_graphmp

##### model configuration
shift_pointer_value=1
apply_tgt_actnode_masks=0
tgt_vocab_masks=1
share_decoder_embed=0

arch=transformer_tgt_pointer_graphmp
tgt_graph_layers="0 1 2"
tgt_graph_heads=2
tgt_graph_mask="allprev_1in1out"

pointer_dist_decoder_selfattn_layers="5"
pointer_dist_decoder_selfattn_heads=1
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=5

apply_tgt_src_align=1
tgt_src_align_layers="0 1 2 3 4 5"
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

seed=42
MAX_EPOCH=120
eval_init_epoch=81

# AUTO NAMING <-- Avoidable?
##### set the experiment dir name based on model configurations

if [[ $pointer_dist_decoder_selfattn_layers == "0 1 2 3 4 5" ]]; then
    lay="all"
else
    lay=""
    for n in $pointer_dist_decoder_selfattn_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'pointer_dist_decoder_selfattn_layers' input: $pointer_dist_decoder_selfattn_layers" && exit 1
        lay=$lay$(( $n + 1 ))
    done
fi

if [[ $tgt_graph_layers == "0 1 2 3 4 5" ]]; then
    grh_lay="all"
else
    grh_lay=""
    for n in $tgt_graph_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'tgt_graph_layers' input: $tgt_graph_layers" && exit 1
        grh_lay=$grh_lay$(( $n + 1 ))
    done
fi

if [[ $tgt_src_align_layers == "0 1 2 3 4 5" ]]; then
    cam_lay="all"
else
    cam_lay=""
    for n in $tgt_src_align_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'tgt_src_align_layers' input: $tgt_src_align_layers" && exit 1
        cam_lay=$cam_lay$(( $n + 1 ))
    done
fi

grh_mask=-$tgt_graph_mask

if [[ $tgt_src_align_focus == "p0c1n0" ]]; then
    cam_focus=""    # default
elif [[ $tgt_src_align_focus == "p0c1n0 p0c0n*" ]]; then
    cam_focus=-abuf    # alignment and "buffer"
fi

# set the experiment directory name
expdir=exp_${features_tag}_act-pos-grh_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

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

# graph structure mask on the decoder self-attention
grh_tag=_grh-lay${grh_lay}-h${tgt_graph_heads}${grh_mask}

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

# combine different model configuration tags to the name
model_tag=${expdir}${ptr_tag}${grh_tag}${cam_tag}${tis_tag}

# All data in this step under
MODEL_FOLDER=DATA/$TASK_TAG/models/$model_tag/ep${MAX_EPOCH}

###############################################################
# ENTITY LINKING
###############################################################

# Smatch evaluation with wiki
BLINK_CACHE_PATH=DATA/EL/BLINK/linkcache

###############################################################
# TESTS 
###############################################################

##### decoding configuration for the final model
BATCH_SIZE=128
BEAM_SIZE=10
DECODING_CHECKPOINT=checkpoint_best_SMATCH.pt
