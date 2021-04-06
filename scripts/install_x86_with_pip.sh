set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset

# This assumes python 3.6.9

# pre-install modules with pip
pip install -r scripts/requirements.txt
pip install --editable .

# install alignment tools
# bash preprocess/install_alignment_tools.sh

# sanity check
python tests/correctly_installed.py
