set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# set data to be used
DATA=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/
input_file=${DATA}/oracles/o3+Word100/dev.en
# reference file
AMR_DEV_FILE=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned

# Set model to be used
# this does not work
features_folder=${DATA}/features/qaldlarge_extracted/
checkpoints_dir=${DATA}/models/stack_transformer_6x6_nopos-qaldlarge_prepro_o3+Word100-stnp6x6-seed42/
# this works
# features_folder=${DATA}/features/o3+Word100_RoBERTa-large-top8/
# checkpoints_dir=${DATA}/models/o3+Word100_RoBERTa-large-top8_stops6x6-seed42/

# folder where we write data
mkdir -p TMP

# run decoding
fairseq-generate \
    $features_folder \
    --gen-subset test \
    --machine-type AMR \
    --machine-rules $checkpoints_dir/train.rules.json \
    --model-overrides "{'pretrained_embed_dim':1024}" \
    --beam 1 \
    --batch-size 10 \
    --path $checkpoints_dir/checkpoint89.pt \
    --results-path TMP/valid

# Create the AMR from the model obtained actions
amr-fake-parse \
    --in-sentences $input_file \
    --in-actions TMP/valid.actions \
    --out-amr TMP/valid.amr \

# Compute score
smatch.py \
     --significant 4  \
     -f $AMR_DEV_FILE \
     TMP/valid.amr \
     -r 10 \
     > TMP/valid.smatch

cat TMP/valid.smatch
