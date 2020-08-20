set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# use this environment for debugging (comment line above)
# eval "$(/path/to/ppc64/miniconda3/bin/conda shell.bash hook)"
# rm -Rf ./tmp_debug
# conda create -y -p ./tmp_debug
# conda activate ./tmp_debug

# install python version to be used
conda install python=3.6.9 -y -c powerai

# pre-install modules with conda 
conda env update -f scripts/stack-transformer/ccc_ppc_fairseq.yml

# fairseq
[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
cd fairseq
git checkout v0.3.0/decouple-fairseq
pip install --editable .
cd ..

# transition_amr_parser
pip install ipdb
pip install --no-deps --editable . 

# install pytorch scatter
rm -Rf pytorch_scatter.ppc
git clone https://github.com/rusty1s/pytorch_scatter.git pytorch_scatter.ppc
cd pytorch_scatter.ppc
git checkout 1.3.2
# Ensure modern GCC
export GCC_DIR=/opt/share/gcc-5.4.0/ppc64le/
export PATH=/opt/share/cuda-9.0/ppc64le/bin:$GCC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
python setup.py develop
cd ..

# smatch v1.0.4
[ ! -d smatch ] && git clone https://github.com/snowblink14/smatch.git smatch
cd smatch
git checkout v1.0.4
cd ..
pip install smatch/
