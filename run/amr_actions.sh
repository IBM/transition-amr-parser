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

# We will need this to save the alignment log
mkdir -p $ORACLE_FOLDER

##### ORACLE EXTRACTION
# Given sentence and aligned AMR, provide action sequence that generates the AMR back
if [ -f $ORACLE_FOLDER/.done ]; then

    printf "[\033[92m done \033[0m] $ORACLE_FOLDER/.done\n"

else

    mkdir -p $ORACLE_FOLDER

    # copy the original AMR data: no wikification
    # cp $ALIGNED_FOLDER/train.txt $ORACLE_FOLDER/ref_train.amr
    # cp $ALIGNED_FOLDER/dev.txt $ORACLE_FOLDER/ref_dev.amr
    # cp $ALIGNED_FOLDER/test.txt $ORACLE_FOLDER/ref_test.amr
    # copy alignment probabilities (if provided)
    [ -f "$ALIGNED_FOLDER/alignment.trn.pretty" ] \
        && cp $ALIGNED_FOLDER/alignment.trn.pretty $ORACLE_FOLDER/
    [ -f "$ALIGNED_FOLDER/alignment.trn.align_dist.npy" ] \
        && cp $ALIGNED_FOLDER/alignment.trn.align_dist.npy $ORACLE_FOLDER/

    echo -e "\nTrain data"
    if [ $MODE = "doc" ] || [ $MODE = "doc+sen" ] || [ $MODE = "doc+sen+pkd" ];then
        train_force_args="--out-fdec-actions ${ORACLE_FOLDER}/train.force_actions"
        if [ $TRAIN_DOC == "conll" ];then
            cp $CONLL_DATA $ORACLE_FOLDER/
            conllf_basename="$(basename $CONLL_DATA)"
            TRAIN_IN_AMR=$ORACLE_FOLDER/${conllf_basename}
            if [ $TRAIN_DOC == "conll+gold" ];then
                echo "making docamrs"
                python scripts/doc-amr/get_doc_amr_from_sen.py \
                    --in-amr $AMR_TRAIN_FILE \
                    --coref-fof $TRAIN_COREF \
                    --fof-path $FOF_PATH \
                    --norm $NORM \
                    --out-amr $ORACLE_FOLDER/train_${NORM}.docamr 
            
            
                cat $TRAIN_IN_AMR >> $ORACLE_FOLDER/train_${NORM}.docamr
                TRAIN_IN_AMR=$ORACLE_FOLDER/train_${NORM}.docamr
            fi


        else
            echo "Doc Mode , making docamrs"
            python scripts/doc-amr/get_doc_amr_from_sen.py \
                --in-amr $AMR_TRAIN_FILE \
                --coref-fof $TRAIN_COREF \
                --fof-path $FOF_PATH \
                --norm $NORM \
                --out-amr $ORACLE_FOLDER/train_${NORM}.docamr 
            
            TRAIN_IN_AMR=$ORACLE_FOLDER/train_${NORM}.docamr
        fi
	    if [ $TRAIN_DOC == "both" ];then
		echo -e "\n Adding conll data"
		cp $CONLL_DATA $ORACLE_FOLDER/
        conllf_basename="$(basename $CONLL_DATA)"
		python src/transition_amr_parser/add_sentence_amrs_to_file.py \
                       --in-amr $ORACLE_FOLDER/${conllf_basename} \
                       --out-amr $TRAIN_IN_AMR
	    
        fi

        if [ $MODE == "doc+sen" ];then
            echo -e "\n Doc+sen mode Adding sentence data"
            python src/transition_amr_parser/add_sentence_amrs_to_file.py \
                --in-amr $AMR_SENT_TRAIN_FILE \
                --out-amr $TRAIN_IN_AMR 
        fi

        if [ $MODE = "doc+sen+pkd" ]; then
            echo -e "\n Doc+sen mode Adding sentence data"
            python src/transition_amr_parser/add_sentence_amrs_to_file.py \
                --in-amr $AMR_SENT_TRAIN_FILE \
                --out-amr $TRAIN_IN_AMR 
            python scripts/doc-amr/pack_amrs.py \
            --in-amr $AMR_SENT_TRAIN_FILE \
            --out-amr $ORACLE_FOLDER/train_sen_packed.amr
            cat $ORACLE_FOLDER/train_sen_packed.amr >> $TRAIN_IN_AMR
        fi
    

    elif [ $MODE == "sen" ];then
        train_force_args=""
        TRAIN_IN_AMR=$AMR_TRAIN_FILE
        cp $TRAIN_IN_AMR $ORACLE_FOLDER/ref_train.amr
    fi

    python src/transition_amr_parser/amr_machine.py \
        --in-aligned-amr $TRAIN_IN_AMR \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-tokens $ORACLE_FOLDER/train.en \
        --absolute-stack-positions \
        --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
        --use-copy ${USE_COPY} \
        $train_force_args \
        $DOC_ORACLE_ARGS
        
        # --reduce-nodes all

    # copy machine config to model config
    for seed in $SEEDS;do
        # define seed and working dir
        checkpoints_dir="${MODEL_FOLDER}seed${seed}/"
	if [ -d $checkpoints_dir ]
	then
        cp $ORACLE_FOLDER/machine_config.json $checkpoints_dir
	fi
    done

    echo -e "\nDev data"
    if [ $MODE == "doc" ] || [ $MODE == "doc+sen+ft" ];then
        echo -e "\n Doc Mode ,Making docamr dev data"
        if [[ $SLIDING == 1 ]];then
            DEV_DOC_ORACLE_ARGS=""
        else
            echo "Truncating dev data since sliding mode is not turned on"
            DEV_DOC_ORACLE_ARGS="--truncate"
        fi
        dev_force_args="--out-fdec-actions ${ORACLE_FOLDER}/dev.force_actions"
        python scripts/doc-amr/get_doc_amr_from_sen.py \
            --in-amr $AMR_DEV_FILE \
            --coref-fof $DEV_COREF \
            --fof-path $FOF_PATH \
            --norm $NORM \
            --out-amr $ORACLE_FOLDER/dev_${NORM}.docamr
        DEV_IN_AMR=$ORACLE_FOLDER/dev_${NORM}.docamr

        echo -e "\n Getting docAMR rep of dev data "
        doc-amr --amr3-path $FOF_PATH \
            --coref-fof $DEV_COREF \
            --out-amr $ORACLE_FOLDER/dev_docAMR.docamr \
            --rep docAMR

    elif [ $MODE == "doc+sen" ] || [ $MODE == "doc+sen+pkd" ];then
        dev_force_args="--out-fdec-actions ${ORACLE_FOLDER}/dev.force_actions"
        if [[ $SLIDING == 1 ]];then
            DEV_DOC_ORACLE_ARGS=""
        else
            echo "Truncating dev data since sliding mode is not turned on"
            DEV_DOC_ORACLE_ARGS="--truncate"
        fi
        if [[ $DEV_CHOICE=="doc" ]];then
            echo -e "\n Doc+sen mode ,dev choice is doc dev, Making docamr dev data to use instead of senamr test in doc+sen mode"
            python scripts/doc-amr/get_doc_amr_from_sen.py \
                --in-amr $AMR_DEV_FILE \
                --coref-fof $DEV_COREF \
                --fof-path $FOF_PATH \
                --norm $NORM \
                --out-amr $ORACLE_FOLDER/dev_${NORM}.docamr
            DEV_IN_AMR=$ORACLE_FOLDER/dev_${NORM}.docamr
            echo -e "\n Getting docAMR rep of docdev data "
            doc-amr --amr3-path $FOF_PATH \
                --coref-fof $DEV_COREF \
                --out-amr $ORACLE_FOLDER/dev_docAMR.docamr \
                --rep docAMR
        else
            echo -e "\n Doc+sen mode , Using sentence dev data"
            DEV_IN_AMR=$AMR_SENT_DEV_FILE
        fi

    elif [ $MODE == "sen" ];then
        DEV_DOC_ORACLE_ARGS=""
        dev_force_args=""
        DEV_IN_AMR=$AMR_DEV_FILE
        cp $DEV_IN_AMR $ORACLE_FOLDER/ref_dev.amr
    fi

    python src/transition_amr_parser/amr_machine.py \
        --in-aligned-amr $DEV_IN_AMR \
        --in-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --out-tokens $ORACLE_FOLDER/dev.en \
        $DEV_DOC_ORACLE_ARGS \
        $dev_force_args

    echo -e "\nTest data"

    if [ $MODE = "doc" ] || [ $MODE = "doc+sen+ft" ];then
        echo -e "\n Doc Mode,Making docamr test data"
        if [[ $SLIDING == 1 ]];then
            TEST_DOC_ORACLE_ARGS=""
        else
            echo "Truncating test data since sliding mode is not turned on"
            TEST_DOC_ORACLE_ARGS="--truncate"
        fi
        test_force_args="--out-fdec-actions ${ORACLE_FOLDER}/test.force_actions"
        python scripts/doc-amr/get_doc_amr_from_sen.py \
            --in-amr $AMR_TEST_FILE \
            --coref-fof $TEST_COREF \
            --fof-path $FOF_PATH \
            --norm $NORM \
            --out-amr $ORACLE_FOLDER/test_${NORM}.docamr

        TEST_IN_AMR=$ORACLE_FOLDER/test_${NORM}.docamr
        echo -e "\n Getting docAMR rep of test data "
        doc-amr --amr3-path $FOF_PATH \
            --coref-fof $TEST_COREF \
            --out-amr $ORACLE_FOLDER/test_docAMR.docamr \
            --rep docAMR
    fi
    
    
    if [ $MODE = "doc+sen" ] || [ $MODE == "doc+sen+pkd" ];then
        if [[ $SLIDING == 1 ]];then
            TEST_DOC_ORACLE_ARGS=""
        else
            echo "Truncating test data since sliding mode is not turned on"
            TEST_DOC_ORACLE_ARGS="--truncate"
        fi
        test_force_args="--out-fdec-actions ${ORACLE_FOLDER}/test.force_actions"
        if [[ $DEV_CHOICE=="doc" ]];then
            echo -e "\n Doc+sen Mode,dev choice is doc . Making docamr test data"
            python scripts/doc-amr/get_doc_amr_from_sen.py \
                --in-amr $AMR_TEST_FILE \
                --coref-fof $TEST_COREF \
                --fof-path $FOF_PATH \
                --norm $NORM \
                --out-amr $ORACLE_FOLDER/test_${NORM}.docamr

            TEST_IN_AMR=$ORACLE_FOLDER/test_${NORM}.docamr
            echo -e "\n Getting docAMR rep of test data "
            doc-amr --amr3-path $FOF_PATH \
                --coref-fof $TEST_COREF \
                --out-amr $ORACLE_FOLDER/test_docAMR.docamr \
                --rep docAMR
        else
            echo -e "\n Doc+sen mode .Using sentence test data"
            TEST_IN_AMR=$AMR_SENT_TEST_FILE
        fi
        
    fi
    
    
    if [ $MODE = "sen" ];then
        TEST_DOC_ORACLE_ARGS=""
        test_force_args=""
        TEST_IN_AMR=$AMR_TEST_FILE
        cp $TEST_IN_AMR $ORACLE_FOLDER/ref_test.amr
    fi

    python src/transition_amr_parser/amr_machine.py \
        --in-aligned-amr $TEST_IN_AMR \
        --in-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/test.actions \
        --out-tokens $ORACLE_FOLDER/test.en \
        $TEST_DOC_ORACLE_ARGS \
        $test_force_args

    touch $ORACLE_FOLDER/.done

fi
