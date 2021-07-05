set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install --editable .

# Pytorch scatter
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
git checkout 1.3.2
cd ..
pip install --editable pytorch_scatter/

# sanity check
python tests/correctly_installed.py
