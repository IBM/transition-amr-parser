#!/bin/bash

set -o errexit
set -o pipefail
# . set_environment.sh
. set_exps.sh    # general setup for experiments management (save dir, etc.)
set -o nounset

# set root directory
# (the config script should be called within other scripts, where data root dir is defined)
if [ -z ${ROOTDIR+x} ]; then
    ROOTDIR=EXP
fi


##############################################################################                                                                                                                         
# DATA                                                                                                                                                                                                 
##############################################################################                                                                                                                        

# Original AMR files in PENMAN notation                                                                                                                                                                
# see preprocess/README.md to create these from LDC folders                                                                                                                                            
# This step will be ignored if the aligned train file below exists                                                                                                                                     

# Example AMR2.0 AMR1.0 dep-parsing CFG                                                                                                                                                                
TASK_TAG=AMR2.0
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



# NOTE: original dev set also same as below
# AMR_DEV_FILE_WIKI=$LDC2016_AMR_CORPUS/dev.txt

##### CONFIG
TASK=amr_action_pointer_bart

ORACLEDIR=data/o10_act-states
EMBDIR=data/en_embeddings

ORACLE_FOLDER=$ROOTDIR/$ORACLEDIR/oracle            # oracle actions, etc.
DATA_FOLDER=$ROOTDIR/$ORACLEDIR/processed           # preprocessed actions states information, etc.
EMB_FOLDER=$ROOTDIR/$EMBDIR/bart_base     # pre-stored pretrained en embeddings (not changing with oracle)

# ENTITIES_WITH_PREDS="person,thing,government-organization,have-org-role-91,monetary-quantity"

# PRETRAINED_EMBED=roberta.base
# PRETRAINED_EMBED_DIM=768
# BERT_LAYERS=12

PRETRAINED_EMBED=bart.base
PRETRAINED_EMBED_DIM=768
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12"
