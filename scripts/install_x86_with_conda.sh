set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# install python version to be used
conda install python=3.7 -y 

# pre-install modules with conda 
conda install pytorch=1.3.0 -y -c pytorch
conda install -c conda-forge nvidia-apex -y

# download fairseq
pip install --editable .

# install alignment tools
# bash preprocess/install_alignment_tools.sh

# sanity check
python tests/correctly_installed.py
