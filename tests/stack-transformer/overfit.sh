# Run small train / test on 25 sentences using same data for train/dev. Note
# that the model fails to overfit (probably due to hyperparameters or data
# size)
set -o errexit
set -o pipefail
if [ -z "$1" ];then
    amr_file=DATA/wiki25.jkaln
else
    amr_file=$1
fi
. set_environment.sh
set -o nounset

experiment_tag=$(basename $amr_file)

# Config
ORACLE_FOLDER=DATA.tests/oracles/$experiment_tag/
FEATURES_FOLDER=DATA.tests/features/$experiment_tag/
MODEL_FOLDER=DATA.tests/models/$experiment_tag/
RESULTS_FOLDER=$MODEL_FOLDER/beam1
max_epoch=10

# ORACLE EXTRACTION
# Given sentence and aligned AMR, provide action sequence that generates the
# AMR back
mkdir -p $ORACLE_FOLDER
python transition_amr_parser/o3_data_oracle.py \
    --in-amr DATA/wiki25.jkaln \
    --out-sentences $ORACLE_FOLDER/train.en \
    --out-actions $ORACLE_FOLDER/train.actions \
    --out-rule-stats $ORACLE_FOLDER/train.rules.json \
    --copy-lemma-action
# For this unit test train/dev/test is the same (memorize test)
cp $ORACLE_FOLDER/train.en $ORACLE_FOLDER/dev.en
cp $ORACLE_FOLDER/train.actions $ORACLE_FOLDER/dev.actions
cp $ORACLE_FOLDER/train.en $ORACLE_FOLDER/test.en
cp $ORACLE_FOLDER/train.actions $ORACLE_FOLDER/test.actions

# PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq
# format
rm -Rf $FEATURES_FOLDER  # not as var for security
mkdir -p $FEATURES_FOLDER
fairseq-preprocess \
    --source-lang en \
    --target-lang actions \
    --trainpref $ORACLE_FOLDER/train \

# PREPROCESSING
# Extract sentence featrures and action sequence and store them in fairseq
# format
rm -Rf DATA.tests/features/$experiment_tag/  # not as var for security
mkdir -p $FEATURES_FOLDER
fairseq-preprocess \
    --source-lang en \
    --target-lang actions \
    --trainpref $ORACLE_FOLDER/train \
    --validpref $ORACLE_FOLDER/dev \
    --testpref $ORACLE_FOLDER/test \
    --destdir $FEATURES_FOLDER \
    --workers 1 \
    --machine-type AMR \
    --entity-rules $entity_rules \
    --machine-rules $ORACLE_FOLDER/train.rules.json 
 
# TRAINING
rm -Rf MODEL_FOLDER
fairseq-train \
    $FEATURES_FOLDER \
    --user-dir ../../transition_amr_parser \
    --max-epoch $max_epoch \
    --burnthrough 5 \
    --arch $model_arch \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1 \
    --pretrained-embed-dim 768 \
    --lr 0.025 \
    --min-lr 1e-09 \
    --dropout 0.0 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.01 \
    --keep-last-epochs 1 \
    --max-tokens 3584 \
    --log-format json \
    --fp16 \
    --seed 42 \
    --save-dir $MODEL_FOLDER

# DECODING
rm -Rf $RESULTS_FOLDER
mkdir -p $RESULTS_FOLDER
# --nbest 3 \
fairseq-generate \
    $FEATURES_FOLDER  \
    --gen-subset valid \
    --machine-type AMR  \
    --entity-rules $entity_rules \
    --machine-rules $ORACLE_FOLDER/train.rules.json \
    --beam 1 \
    --batch-size 15 \
    --remove-bpe \
    --path $MODEL_FOLDER/checkpoint${max_epoch}.pt  \
    --results-path $RESULTS_FOLDER/valid \

# Create the AMR from the model obtained actions
python transition_amr_parser/o3_fake_parse.py \
    --in-sentences $ORACLE_FOLDER/dev.en \
    --in-actions $RESULTS_FOLDER/valid.actions \
    --out-amr $RESULTS_FOLDER/valid.amr \

# Smatch evaluation without wiki
smatch.py \
     --significant 4  \
     -f $amr_file \
     $RESULTS_FOLDER/valid.amr \
     -r 10 \
     > $RESULTS_FOLDER/valid.smatch

cat $RESULTS_FOLDER/valid.smatch

if [ "$(cat $RESULTS_FOLDER/valid.smatch)" != "F-score: 0.1592" ];then
        echo -e "[\033[91mFAILED\033[0m] overfitting test"
        exit 1
else
        echo -e "[\033[92mOK\033[0m] overfitting test"
fi
