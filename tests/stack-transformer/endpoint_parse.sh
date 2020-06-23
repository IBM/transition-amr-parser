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
input_file=${DATA}/oracles/o3+Word100/dev.en
# reference file
AMR_DEV_FILE=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned

# Set model to be used
# features_folder=${DATA}/features/qaldlarge_extracted/
# checkpoints_dir=${DATA}/models/stack_transformer_6x6_nopos-qaldlarge_prepro_o3+Word100-stnp6x6-seed42/
features_folder=${DATA}/features/o3+Word100_RoBERTa-large-top8/
checkpoints_dir=${DATA}/models/o3+Word100_RoBERTa-large-top8_stops6x6-seed42/

# folder where we write data
mkdir -p TMP

# run decoding
# kernprof -l scripts/stack-transformer/parse.py \
python scripts/stack-transformer/parse.py \
    $features_folder \
    --source-lang en \
    --target-lang actions \
    --path $checkpoints_dir/checkpoint89.pt \
    --model-overrides "{'pretrained_embed_dim':1024}" \
    --pretrained-embed roberta.large \
    --bert-layers 17 18 19 20 21 22 23 24 \
    --input $input_file \
    --machine-type AMR \
    --machine-rules $checkpoints_dir/train.rules.json \
    --roberta_batch_size 10 \
    --batch-size 10
# python -m line_profiler parse.py.lprof

# FIXME: removed for debugging
#    --roberta-cache-path ./cache/roberta.large \
