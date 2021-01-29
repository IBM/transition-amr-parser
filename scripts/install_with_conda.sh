set -o errexit
set -o pipefail
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh

### set local environment
cenv_name=amr0.4_draft

[ ! -d $cenv_name ] && echo "creating local conda env ./$cenv_name" && conda create -y -p ./$cenv_name python=3.7

if [[ -z $CONDA_DEFAULT_ENV || $(basename $CONDA_DEFAULT_ENV) != $cenv_name ]]; then
    echo "activating conda environment ./$cenv_name"
    source activate ./$cenv_name
    # or conda activate ./$cenv_name    for newer version of conda
fi  

set -o nounset

# install python version to be used
# conda install python=3.7 -y

# pre-install modules with conda
conda install pytorch=1.2 -y -c pytorch
# [optional] apex
# conda install -c conda-forge nvidia-apex -y

# fairseq
pip install fairseq==0.8.0

# install tensorboardX
pip install tensorboardX
# install packaging for compatibility of PyTorch < 1.2
conda install packaging -y

# install transition_amr_parser
pip install spacy ipdb
# install repo
pip install --editable .

# install alignment tools
# bash preprocess/install_alignment_tools.sh

# install pytorch_scatter
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
git checkout 1.3.2
python setup.py develop
cd ..

# install smatch v1.0.4
[ ! -d smatch ] && git clone https://github.com/snowblink14/smatch.git smatch
cd smatch
git checkout v1.0.4
cd ..
pip install smatch/
## if the above doesn't work with due to an EnvironmentError: [Errno 13]
# python setup.py install
# cd ..


# sanity check
python tests/correctly_installed.py
