set -o errexit
set -o pipefail 

# activate conda
# FIXME: Replace this with your conda
eval "$(/nobackup/users/ramast/miniconda3/bin/conda shell.bash hook)"
# Create local env if missing
[ ! -d cenv_ppc ] && conda create -y -p ./cenv_ppc
echo "conda activate ./cenv_ppc"
conda activate ./cenv_ppc

# accept POWER AI license
export IBM_POWERAI_LICENSE_ACCEPT=yes

# this may not be needed
export PYTHONPATH=.

set -o nounset 

# install python version to be used
conda install -y pytorch==1.4.0 -c pytorch -c powerai

# fairseq
[ ! -d fairseq ] && git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout v0.10.2
pip install --editable .
cd ..

# smatch
[ ! -d smatch ] && git clone https://github.com/snowblink14/smatch.git
cd smatch
git checkout v1.0.4
pip install .
cd ..

# repo instal proper
pip install --editable .

# TODO: Install pytorch scatter

# Tried to use this, but gcc is not available to load

# module load gcc
# pip install torch-scatter --no-cache-dir

# This is what I did IBM's CCC PPC machines. Bottom line we need a GCC higher
# than the one available by default

# # install pytorch scatter
# rm -Rf pytorch_scatter.ppc
# git clone https://github.com/rusty1s/pytorch_scatter.git pytorch_scatter.ppc
# cd pytorch_scatter.ppc
# git checkout 1.3.2
# Ensure modern GCC
#export GCC_DIR=/opt/share/gcc-5.4.0/ppc64le/
#export PATH=/opt/share/cuda-9.0/ppc64le/bin:$GCC_DIR/bin:$PATH
#export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
#python setup.py develop
#cd ..

# check all ok
python tests/correctly_installed.py
