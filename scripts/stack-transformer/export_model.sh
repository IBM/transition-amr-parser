set -o errexit 
set -o pipefail
# setup environment
# . set_environment.sh
# Argument handling
model_folder=$1
[ -z "$model_folder" ] && \
    echo -e "\n$0 <model_folder\n" && \
    exit 1
set -o nounset 

EXTERNAL_FOLDER=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/
# FIXME: We need this to be a global variable in configs. Right now DATA/AMR is
# hardcoded here
LOCAL_FOLDER=DATA/AMR/

# Load config
[ ! -f "$model_folder/config.sh" ] && \
    echo "Expected $model_folder/config.sh" && \
    exit 1
. "$model_folder/config.sh"

# name of folder where checlpoints are
model_folder_name=$(basename $model_folder)

# Check of model exists in remote
target_oracle_folder="$EXTERNAL_FOLDER/oracles/${ORACLE_TAG}/"
target_features_folder="$EXTERNAL_FOLDER/features/${ORACLE_TAG}_${PREPRO_TAG}/"
target_checkpoints_dir_root="$EXTERNAL_FOLDER/models/$model_folder_name"

features_folder=$LOCAL_FOLDER/features/${ORACLE_TAG}_${PREPRO_TAG}/

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
