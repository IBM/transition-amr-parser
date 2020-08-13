set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset 

# use this environment for debugging (comment line above)
# eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
# rm -Rf ./tmp_debug
# conda create -y -p ./tmp_debug
# conda activate ./tmp_debug

# fairseq
[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
conda env update -f scripts/stack-transformer/ccc_x86_fairseq.yml
cd fairseq
# TODO: Copy parsing data
# TODO: Copy transition_based_parsing task
git checkout v0.3.0/decouple-fairseq
pip install --editable .
cd ..

# transition_amr_parser
# install not previously installed dependencies with conda or pip when not
# possible
pip install spacy ipdb
pip install --editable . 

# install pytorch scatter
rm -Rf  pytorch_scatter.x86
git clone https://github.com/rusty1s/pytorch_scatter.git pytorch_scatter.x86
cd pytorch_scatter.x86
git checkout 1.3.2
# Ensure modern GCC
export GCC_DIR=/opt/share/gcc-5.4.0/x86_64/
export PATH=/opt/share/cuda-9.0/x86_64/bin:$GCC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
python setup.py develop
cd ..

# smatch
[ ! -d smatch ] && git clone git@github.ibm.com:mnlp/smatch.git
cd smatch
git checkout f728c3d3f4a71b44678224d6934c1e67c4d37b89
cd ..
pip install smatch/
