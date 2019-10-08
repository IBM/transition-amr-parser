set -o errexit
set -o pipefail 
#set -o nounset 

# Check lauched from correct folder
[ ! -d 'scripts' ] && echo "Launch as scripts/$0" && exit
# Check this is a PowerPC machine
[[ "$HOSTNAME" =~ dccpc.* ]] || \
    { echo >&2 "Not PowerPC. Aborting."; exit 1; }

# PowerPC installation use conda 
command -v conda >/dev/null 2>&1 || \
    { echo >&2 "Need conda. Aborting."; exit 1; }

# We need to create an individual environment
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
env_name=$(basename $(pwd))
# Remove
conda env remove -n $env_name
# normal env
conda env create -n $env_name -f scripts/stack_lstm_amr.yml 
conda activate $env_name

# spacy lemmatization
python -m spacy download en

# smatch
git clone https://github.com/snowblink14/smatch.git
