set -o errexit
set -o pipefail 
#set -o nounset 

[ -z "$CONDA_PREFIX" ] && echo "Expecting environment variable CONDA_PREFIX" && exit 1

# We need to create an individual environment
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
env_name=$(basename $(pwd))
# Remove
conda env remove -n $env_name
# normal env

# FIXME: Installing apex leads to FusedAdam error (missing parameter).
# Commented for now.
sed 's@ *- apex@# &@' -i fairseq/dcc/ccc_pcc_fairseq.yml

[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
conda env create -n $env_name -f fairseq/dcc/ccc_pcc_fairseq.yml 
conda activate $env_name

# install transition_amr_parser
sed 's@^@# @' -i requirements.txt
pip install --editable .
git checkout requirements.txt

# install fairseq
pip install --editable fairseq/

# install pytorch scatter
[ ! -d pytorch_scatter ] && git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
conda install -y pytest-runner -c powerai
# Ensure modern GCC
export GCC_DIR=/opt/share/gcc-5.4.0/ppc64le/
export PATH=/opt/share/cuda-9.0/ppc64le/bin:$GCC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
python setup.py develop
cd ..

# Install smatch, use MNLPs smatch (importable and faster) 
rm -Rf smatch
git clone git@github.ibm.com:mnlp/smatch.git
# editable just in case
pip install --editable smatch/
