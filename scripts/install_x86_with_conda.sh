set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# install python version to be used
conda install python=3.6.9 -y 

# pre-install modules with conda 
conda install pytorch=1.1.0 -y
conda install -c conda-forge nvidia-apex -y

# download fairseq
# public branch and patch
bash scripts/download_and_patch_fairseq.sh
# private branch (for development)
# git clone git@github.ibm.com:ramon-astudillo/fairseq.git fairseq-stack-transformer
# cd fairseq-stack-transformer
# git checkout v0.3.0/decouple-fairseq
# cd ..

# install repos
pip install --no-deps --editable fairseq-stack-transformer
pip install --editable .

# install alignment tools
# bash preprocess/install_alignment_tools.sh

# sanity check
python tests/correctly_installed.py
