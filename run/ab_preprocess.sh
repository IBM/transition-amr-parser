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

# Require aligned AMR
[ ! -f "$ORACLE_FOLDER/.done" ] && \
    echo -e "\nRequires oracle in $ORACLE_FOLDER\n" && \
    exit 1

# TODO: Recover support of fine-tuning mode
# # Dictionary update for fine-tuning. We will add the words from the fine-tuning
# # vocabulary to the pretrained one. Note that there is a similar if below
# # to adjust pretrained model embeddings accordingly
# if [[ "$FAIRSEQ_TRAIN_ARGS" =~ .*"--restore-file".* ]];then
# 
#     # Work with a copy of the pretrained dictionaries (will be modified)
#     mkdir -p $FEATURES_FOLDER
# 
#     # source 
#     cp $PRETRAINED_SOURCE_DICT ${SRC_DICT}
#     python scripts/create_fairseq_dicts.py \
#         --in-pretrain-dict $SRC_DICT \
#         --in-fine-tune-data $ORACLE_FOLDER/train.en \
#     
#     # target
#     cp $PRETRAINED_TARGET_DICT ${TGT_DICT}
#     python scripts/create_fairseq_dicts.py \
#         --in-pretrain-dict $TGT_DICT \
#         --in-fine-tune-data $ORACLE_FOLDER/train.actions \
# 
# fi

##### PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq format

TASK=${TASK:-amr_action_pointer}

if [[ (-f $DATA_FOLDER/.done) && (-f $EMB_FOLDER/.done) ]]; then

    echo "Directory to processed oracle data: $DATA_FOLDER"
    echo "and source pre-trained embeddings: $EMB_FOLDER"
    echo "already exists --- do nothing."

else

    # If folder exists but its not .done, delete content (otherwise fairseq
    # will complain). Make sure its not an empty string
    [ -d "$DATA_FOLDER" ] && [ ! -z "${DATA_FOLDER// }" ] && [ ! -f $DATA_FOLDER/.done ] && \
        echo "Cleaning up partially completed $DATA_FOLDER" && \
        for file in $(find $DATA_FOLDER -maxdepth 1 -type f);do
            echo "rm $file"
            rm $file
        done

    if [[ $TASK == "amr_action_pointer" ]]; then

        python fairseq_ext/preprocess.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS
        #     --machine-type AMR \
        #     --machine-rules $ORACLE_FOLDER/train.rules.json

    elif [[ $TASK == "amr_action_pointer_graphmp" ]]; then

        python fairseq_ext/preprocess_graphmp.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS

    elif [[ $TASK == "amr_action_pointer_graphmp_amr1" ]]; then

        # a separate route of code for preprocessing of AMR 1.0 data; the only
        # difference is in o8 state machine
        # get_valid_canonical_actions to deal with a single exmple in training
        # set with self-loop
    
        python fairseq_ext/preprocess_graphmp.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS

    elif [[ $TASK == "amr_action_pointer_bart" ]]; then

        python fairseq_ext/preprocess_bart.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
	    --thresholdtgt $THRESHOLDTGT \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS \
	    --langs $LANGS

    elif [[ $TASK == "amr_action_pointer_bartsv" ]]; then

        python fairseq_ext/preprocess_bartsv.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --node-freq-min ${NODE_FREQ_MIN:-5} \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS

    elif [[ $TASK == "amr_action_pointer_bart_dyo" ]]; then

        python fairseq_ext/preprocess_bart.py \
            $FAIRSEQ_PREPROCESS_FINETUNE_ARGS \
            --user-dir ./fairseq_ext \
            --task $TASK \
            --source-lang en \
            --target-lang actions \
            --trainpref $ORACLE_FOLDER/train \
            --validpref $ORACLE_FOLDER/dev \
            --testpref $ORACLE_FOLDER/test \
            --destdir $DATA_FOLDER \
            --embdir $EMB_FOLDER \
            --workers 1 \
            --pretrained-embed $PRETRAINED_EMBED \
            --bert-layers $BERT_LAYERS

    else

        echo -e "\nError: task [$TASK] not recognized\n" && exit 1

    fi

    touch $DATA_FOLDER/.done
    touch $EMB_FOLDER/.done

fi

# TODO: Recover support of fine-tuning
# # In fine-tune mode, we may need to adjust model size
# if [[ "$FAIRSEQ_TRAIN_ARGS" =~ .*"--restore-file".* ]];then
#     # We will modify the checkpoint, so we need to copy it
#     [ ! -f "$RESTORE_FILE" ] && \
#         cp $PRETRAINED_MODEL $RESTORE_FILE
#     python scripts/merge_restored_vocabulary.py $FAIRSEQ_TRAIN_ARGS
# fi
