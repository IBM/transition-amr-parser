set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install --editable .

# sanity check
python tests/correctly_installed.py
