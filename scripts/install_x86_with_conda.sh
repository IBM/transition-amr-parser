set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# install python version to be used
conda install python=3.6.9 -y 

# pre-install modules with conda and pip
conda env update -f scripts/stack-transformer/environment.yml
pip install spacy==2.2.3 smatch==1.0.4 ipdb

# download fairseq
# public branch and patch
bash scripts/download_and_patch_fairseq.sh
# private branch (for development)
# git clone git@github.ibm.com:ramon-astudillo/fairseq.git fairseq-stack-transformer-v0.3.2
# cd fairseq-stack-transformer-v0.3.2
# git checkout v0.3.0/decouple-fairseq
# cd ..

# install repos
pip install --no-deps --editable fairseq-stack-transformer-v0.3.2
pip install --editable .

# install alignment tools
bash preprocess/install_alignment_tools.sh

# sanity check
python tests/correctly_installed.py
