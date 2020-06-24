set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# sanity check
[ ! -f tests/stack-transformer/endpoint_parse.sh ] && \
    echo "Please call this as bash tests/stack-transformer/endpoint_parse.sh" && \
    exit 1

# set data to be used
DATA=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/
ORACLE_TAG=o5+Word100
PREPRO_TAG="RoBERTa-large-top24"
TRAIN_TAG=stnp6x6
# reference file
AMR_DEV_FILE=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned


input_file=${DATA}/oracles/$ORACLE_TAG/dev.en

# Set model to be used
# features_folder=${DATA}/features/qaldlarge_extracted/
# checkpoints_dir=${DATA}/models/stack_transformer_6x6_nopos-qaldlarge_prepro_o3+Word100-stnp6x6-seed42/
features_folder=${DATA}/features/${ORACLE_TAG}_${PREPRO_TAG}
checkpoints_dir=${DATA}/models/${ORACLE_TAG}_${PREPRO_TAG}_${TRAIN_TAG}-seed42/

# folder where we write data
mkdir -p TMP

# TODO: Remove extra arguments, read only folder checkpoint and deduce aregs
# from it
# run decoding
# kernprof -l scripts/stack-transformer/parse.py \
python scripts/stack-transformer/parse.py \
    $features_folder \
    --source-lang en \
    --target-lang actions \
    --path $checkpoints_dir/checkpoint_top3-average_SMATCH.pt \
    --model-overrides "{'pretrained_embed_dim':1024, 'task': 'translation'}" \
    --pretrained-embed roberta.large \
    --bert-layers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
    --input $input_file \
    --machine-type AMR \
    --machine-rules $checkpoints_dir/train.rules.json \
    --roberta_batch_size 10 \
    --batch-size 10

# # python -m line_profiler parse.py.lprof
# 
# # FIXME: removed for debugging
# #    --roberta-cache-path ./cache/roberta.large \

smatch.py \
     --significant 4  \
     -f $AMR_DEV_FILE \
     tmp.amr \
     -r 10 \
     > tmp.smatch

cat tmp.smatch
