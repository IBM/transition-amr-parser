#!/bin/bash

set -o errexit
set -o pipefail

# Argument handling
# First argument must be checkpoint
HELP="\nbash $0 <checkpoint> [-o results_prefix] [-s (dev/test)] [-b beam_size]\n"
[ -z "$1" ] && echo -e "$HELP" && exit 1
first_path=$(echo $1 | sed 's@:.*@@g')
[ ! -f "$first_path" ] && "Missing $1" && exit 1
checkpoint=$1
# process the rest with argument parser
results_prefix=""
data_split2=dev
beam_size=1
shift 
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) results_prefix="$2"; shift 2;;
    -s) data_split2="$2"; shift 2;;
    -b) beam_size="$2"; shift 2;;
    *) echo "unrecognized argument: $1"; exit 1;;
  esac
done

# activate virtualenenv and set other variables
. set_environment.sh

set -o nounset
# extract config from checkpoint path
model_folder=$(dirname $first_path)
config=$model_folder/config.sh
[ ! -f "$config" ] && "Missing $config" && exit 1

# Load config
echo "[Configuration file:]"
echo $config
. $config 


if [ $MODE == "doc" ] || [ $MODE == "doc+sen+ft" ];then
    echo "mode doc"
    echo "using doc amr with rep docAMR as reference amr"
    if [ $data_split2 == "dev" ]; then
        data_split=valid
        data_split_name=dev
        reference_amr=$ORACLE_FOLDER/dev_docAMR.docamr
    elif [ $data_split2 == "test" ]; then
        data_split=test
        data_split_name=test
        reference_amr=$ORACLE_FOLDER/test_docAMR.docamr
    
    else
        echo "$2 is invalid; must be dev or test"
        exit 1
    fi
    if [[ ! -z ${ORACLE_SKIP_ARGS+x} && $ORACLE_SKIP_ARGS =~ "--avoid-indices" ]]; then
                echo "removing indices from reference amr"
                
                param_str="\"|${ORACLE_SKIP_ARGS//[$'\t\r\n']}|\""
                
                python scripts/doc-amr/remove_amrs.py \
                    --in-amr $reference_amr \
                    --arg-str "$param_str" \
                    --out-amr $ORACLE_FOLDER/${data_split_name}_${NORM}_avoid_indices_removed.docamr

                reference_amr=$ORACLE_FOLDER/${data_split_name}_${NORM}_avoid_indices_removed.docamr

                ORACLE_EN=$ORACLE_FOLDER/${data_split_name}_avoid_indices_removed.en
                echo "removing indices from oracle"
                python scripts/doc-amr/remove_sen.py \
                    --in-file $ORACLE_FOLDER/${data_split_name}.en \
                    --arg-str "$param_str" \
                    --out-file $ORACLE_EN
       
    else
        ORACLE_EN=$ORACLE_FOLDER/${data_split_name}.en

    fi
    
    

elif [ $MODE == "doc+sen" ] || [ $MODE == "doc+sen+pkd" ];then
    echo "mode doc+sen"
    if [ $data_split2 == "dev" ]; then
        data_split=valid
        data_split_name=dev
        if [[ $DEV_CHOICE=="doc" ]];then
            echo "dev choice is doc, using doc dev amr as reference amr for dev"
            reference_amr=$ORACLE_FOLDER/dev_docAMR.docamr
        else
            echo "use sen amr as reference amr for dev"
            
        reference_amr=$AMR_SENT_DEV_FILE
        wiki=$LINKER_CACHE_PATH/dev.wiki
        #FIXME
        reference_amr_wiki=$AMR_DEV_FILE_WIKI
        fi

    elif [ $data_split2 == "test" ]; then
        
        data_split=test
        data_split_name=test
        if [[ $DEV_CHOICE=="doc" ]];then
            echo "dev choice is doc, using dev doc amr as reference amr"
            reference_amr=$ORACLE_FOLDER/test_docAMR.docamr
        else
            reference_amr=$AMR_SENT_TEST_FILE
            wiki=$LINKER_CACHE_PATH/test.wiki
            #FIXME
            reference_amr_wiki=$AMR_TEST_FILE_WIKI
        fi
    else
        echo "$2 is invalid; must be dev or test"
        exit 1
    fi
    ORACLE_EN=$ORACLE_FOLDER/${data_split_name}.en


elif [ $MODE == "sen" ];then
    # set data split parameters
    echo "mode sen"
    echo "use sen amr as reference amr"
    if [ $data_split2 == "dev" ]; then
        data_split=valid
        data_split_name=dev
        reference_amr=$AMR_DEV_FILE
        wiki=$LINKER_CACHE_PATH/dev.wiki
        reference_amr_wiki=$AMR_DEV_FILE_WIKI
    elif [ $data_split2 == "test" ]; then
        data_split=test
        data_split_name=test
        reference_amr=$AMR_TEST_FILE
        wiki=$LINKER_CACHE_PATH/test.wiki
        reference_amr_wiki=$AMR_TEST_FILE_WIKI
    else
        echo "$2 is invalid; must be dev or test"
        exit 1
    fi
    ORACLE_EN=$ORACLE_FOLDER/${data_split_name}.en
fi


# we may have to re-compute features
if [[ (-f $DATA_FOLDER/.done) && (-f $EMB_FOLDER/.done) ]]; then
    echo "Using $DATA_FOLDER/"
else
    echo "Re-computing features, may take a while"
    bash run/preprocess.sh $config
fi

RESULTS_FOLDER=$(dirname $first_path)/beam${beam_size}
# Generate results_prefix name if not provided
if [ "$results_prefix" == "" ];then
    results_prefix=$RESULTS_FOLDER/${data_split}_$(basename $first_path)
fi


##### DECODING
mkdir -p $RESULTS_FOLDER
# --nbest 3 \
# --quiet

reg_generate () {
    python src/fairseq_ext/generate.py \
                $DATA_FOLDER  \
                --emb-dir $EMB_FOLDER \
                --user-dir ./fairseq_ext \
                --task $TASK \
                --gen-subset $data_split \
                --src-fix-emb-use $src_fix_emb_use \
                --machine-type AMR  \
                --machine-rules $ORACLE_FOLDER/train.rules.json \
                --machine-config $ORACLE_FOLDER/machine_config.json \
                --modify-arcact-score 1 \
                --use-pred-rules $USE_PRED_RULES \
                --beam $beam_size \
                --batch-size $BATCH_SIZE \
                --remove-bpe \
                --path $checkpoint \
                --quiet \
                --results-path $results_prefix
}

if [ ! -f "${results_prefix}.actions" ];then
    echo "Generating ${results_prefix}.actions"
    if [[ "$MODE" =~ .*"doc".* ]];then
        if [[ $SLIDING == 1 ]]; then
            echo "Sliding mode"
            validarr=($(ls $ORACLE_FOLDER/${data_split_name}_*.en | sed 's/\.en//g'))
            num=${#validarr[@]}
            for i in $(seq 0 $((num-1)) ); do
            cp $ORACLE_FOLDER/${data_split2}_$((i)).force_actions $DATA_FOLDER/${data_split}"_"$((i)).en-actions.force_actions
            cp $DATA_FOLDER/${data_split}_$((i)).en-actions.en $RESULTS_FOLDER/${data_split}_$((i)).en
            done
            gen_subsets=${data_split}_0
            for i in $(seq 1 $((num-1)) ); do gen_subsets=${gen_subsets}","${data_split}_$i ; done;
            python src/fairseq_ext/generate_sliding.py \
                $DATA_FOLDER  \
                --emb-dir $EMB_FOLDER \
                --user-dir ./fairseq_ext \
                --task $TASK \
                --gen-subset $gen_subsets \
                --src-fix-emb-use $src_fix_emb_use \
                --machine-type AMR  \
                --machine-rules $ORACLE_FOLDER/train.rules.json \
                --machine-config $ORACLE_FOLDER/machine_config.json \
                --modify-arcact-score 1 \
                --use-pred-rules $USE_PRED_RULES \
                --beam $beam_size \
                --batch-size $BATCH_SIZE \
                --remove-bpe \
                --path $checkpoint \
                --quiet \
                --results-path $RESULTS_FOLDER/${data_split}
        
            echo "merging all splits"
            cp $DATA_FOLDER/${data_split}.windows $RESULTS_FOLDER/${data_split}.windows
            python src/transition_amr_parser/merge_sliding_splits.py \
            --input-dir $RESULTS_FOLDER \
            --data-split ${data_split}

            cp $RESULTS_FOLDER/${data_split}"_merged.actions" ${results_prefix}".actions"
        
        else
            reg_generate
        fi
    else
        reg_generate
    fi
fi

if [ ! -f "${results_prefix}.amr" ];then
    ##### Create the AMR from the model obtained actions
    python src/transition_amr_parser/amr_machine.py \
        --in-machine-config $ORACLE_FOLDER/machine_config.json \
        --in-tokens $ORACLE_EN \
        --in-actions ${results_prefix}.actions \
        --out-amr ${results_prefix}.amr
        #--in-tokens $ORACLE_FOLDER/${data_split_name}.en \
fi

if [ ! -f "${results_prefix}_docAMR.amr" ];then
    ## Change rep of docamr to docAMR for smatch
    if [ $MODE == "doc" ] || [ $MODE == "doc+sen+ft" ];then
        echo "mode doc"
        echo -e "\n Changing rep of dev/test data to docAMR "
        doc-amr \
            --in-doc-amr-pairwise ${results_prefix}.amr \
            --rep docAMR \
            --pairwise-coref-rel same-as \
            --out-amr ${results_prefix}_docAMR.amr
        results_prefix=${results_prefix}_docAMR
    elif [ $MODE == "doc+sen" ] || [ $MODE == "doc+sen+pkd" ];then
        echo "mode doc+sen"
        if [ $DEV_CHOICE == "doc" ]; then
            echo -e "\n Dev choice is doc, Changing rep of doc data to docAMR norm "
            doc-amr \
                --in-doc-amr-pairwise ${results_prefix}.amr \
                --rep docAMR \
                --pairwise-coref-rel same-as \
                --out-amr ${results_prefix}_docAMR.amr
            results_prefix=${results_prefix}_docAMR
        fi
    fi
fi


# GRAPH POST-PROCESSING

if [ "$LINKER_CACHE_PATH" == "" ];then

    # just copy AMR to wiki AMR
    cp ${results_prefix}.amr ${results_prefix}.wiki.amr

# TODO: Unelegant detection of linker method (temporary)
elif [ -f "${LINKER_CACHE_PATH}/trn.wikis" ];then

    # Legacy linker 
    python scripts/add_wiki.py \
        ${results_prefix}.amr $wiki $LINKER_CACHE_PATH \
        > ${results_prefix}.wiki.amr

else

    # BLINK cache
    python scripts/retyper.py \
        --inputfile ${results_prefix}.amr \
        --outputfile ${results_prefix}.wiki.amr \
        --skipretyper \
        --wikify \
        --blinkcachepath $LINKER_CACHE_PATH \
        --blinkthreshold 0.0

fi


##### SMATCH evaluation
if [[ "$EVAL_METRIC" == "smatch" ]]; then

    # Smatch evaluation without wiki

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.amr > ${results_prefix}.amr.no_isi
    # if [[ "$reference_amr" =~ ".docamr" ]];then
    #     echo "removing isi from reference amr"
    #     sed 's@\~[0-9]\{1,\}@@g' $reference_amr > ${reference_amr}.no_isi
    #     reference_amr=${reference_amr}.no_isi
    # fi
    echo "Computing SMATCH between ---"
    echo "$reference_amr"
    echo "${results_prefix}.amr"

    if [[ $MODE == "doc" || ( $MODE == "doc+sen" && $DEV_CHOICE == "doc" ) || $MODE == "doc+sen+pkd" ]];then
            
            doc-smatch -r 1 --significant 4 --coref-subscore \
                -f $reference_amr \
                ${results_prefix}.amr.no_isi \
                | tee ${results_prefix%"_docAMR"}.smatch
            
    else
        smatch.py -r 10 --significant 4 \
            -f $reference_amr \
            ${results_prefix}.amr.no_isi \
            | tee ${results_prefix}.smatch
    fi

elif [[ "$EVAL_METRIC" == "wiki.smatch" ]]; then

    # Smatch evaluation without wiki

    # until smatch is fixed, we need to remove the ISI alignment annotations
    sed 's@\~[0-9]\{1,\}@@g' ${results_prefix}.wiki.amr > ${results_prefix}.wiki.amr.no_isi
    if [[ "$reference_amr" =~ ".docamr" ]];then
        echo "removing isi from reference amr"
        sed 's@\~[0-9]\{1,\}@@g' $reference_amr > ${reference_amr}.no_isi
        reference_amr=${reference_amr}.no_isi
    fi
    # compute score
    echo "Computing SMATCH between ---"
    echo "$reference_amr_wiki"
    echo "${results_prefix}.wiki.amr"
    smatch.py -r 10 --significant 4  \
         -f $reference_amr_wiki \
         ${results_prefix}.wiki.amr.no_isi \
         | tee ${results_prefix}.wiki.smatch

fi
