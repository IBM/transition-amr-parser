set -o errexit 
set -o pipefail
# setup environment
# . set_environment.sh
# Argument handling
model_folder=$1
[ -z "$model_folder" ] && model_folder=""
set -o nounset 

EXTERNAL_FOLDER=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/

#
[ "$model_folder" == "" ] && \
    python scripts/stack-transformer/rank_model.py --checkpoints $EXTERNAL_FOLDER/models --seed-average

# Check of model exists in remote
target_oracle_folder="$EXTERNAL_FOLDER/oracles/${ORACLE_TAG}/"
target_features_folder="$EXTERNAL_FOLDER/features/${ORACLE_TAG}_${PREPRO_TAG}/"
target_checkpoints_dir_root="$EXTERNAL_FOLDER/models/$model_folder_name"

# FIXME: We need this to be a global variable in configs. Right now DATA/AMR is
# hardcoded here
features_folder=DATA/AMR/features/${ORACLE_TAG}_${PREPRO_TAG}/

# ORACLE
if [ -d "$target_oracle_folder" ];then
    echo "Skiping existing: $target_oracle_folder"
else
    cp -R $ORACLE_FOLDER $EXTERNAL_FOLDER/oracles/
fi

# FEATURES
if [ -d "$target_features_folder" ];then
    echo "Skiping existing: $target_features_folder"
else
    echo "cp -R $features_folder $EXTERNAL_FOLDER/features/"
    cp -R $features_folder $EXTERNAL_FOLDER/features/
fi

# MODEL
if [ -d "$target_checkpoints_dir_root" ];then
    echo "Skiping existing: $target_checkpoints_dir_root"
else
    echo "cp -R $model_folder $target_checkpoints_dir_root"
    cp -R $model_folder $target_checkpoints_dir_root
fi
