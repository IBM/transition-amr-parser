set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# install python version to be used
conda install python=3.6.9 -y -c powerai

# pre-install modules with conda 
# Note spacy only available with conda. Version will not match x86 one
conda env update -f scripts/environment_ppc.yml

# install repos
pip install -r scripts/requirements_ppc.txt
pip install --no-deps --editable .

# install alignment tools
# only use on x86
# bash preprocess/install_alignment_tools.sh

# sanity check
python tests/correctly_installed.py
