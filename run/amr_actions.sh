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
    cp $ALIGNED_FOLDER/train.txt $ORACLE_FOLDER/ref_train.amr
    cp $ALIGNED_FOLDER/dev.txt $ORACLE_FOLDER/ref_dev.amr
    cp $ALIGNED_FOLDER/test.txt $ORACLE_FOLDER/ref_test.amr
    # copy alignment probabilities (if provided)
    [ -f "$ALIGNED_FOLDER/alignment.trn.pretty" ] \
        && cp $ALIGNED_FOLDER/alignment.trn.pretty $ORACLE_FOLDER/
    [ -f "$ALIGNED_FOLDER/alignment.trn.align_dist.npy" ] \
        && cp $ALIGNED_FOLDER/alignment.trn.align_dist.npy $ORACLE_FOLDER/

    echo -e "\nTrain data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TRAIN_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/train.actions \
        --out-tokens $ORACLE_FOLDER/train.en \
        --absolute-stack-positions  \
        --out-stats-vocab $ORACLE_FOLDER/train.actions.vocab \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    # copy machine config to model config
    for seed in $SEEDS;do
        # define seed and working dir
        checkpoints_dir="${MODEL_FOLDER}-seed${seed}/"
        cp $ORACLE_FOLDER/machine_config.json $checkpoints_dir
    done

    echo -e "\nDev data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_DEV_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/dev.actions \
        --out-tokens $ORACLE_FOLDER/dev.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    echo -e "\nTest data"

    python transition_amr_parser/amr_machine.py \
        --in-aligned-amr $AMR_TEST_FILE \
        --out-machine-config $ORACLE_FOLDER/machine_config.json \
        --out-actions $ORACLE_FOLDER/test.actions \
        --out-tokens $ORACLE_FOLDER/test.en \
        --absolute-stack-positions  \
        --use-copy ${USE_COPY} \
        # --reduce-nodes all

    touch $ORACLE_FOLDER/.done

fi
