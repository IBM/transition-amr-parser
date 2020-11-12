set -o errexit
set -o pipefail

### set environment

### default shared env
# conda create -y -n amr0.4
# source activate amr0.4

### local env
# cenv_name=amr0.4_ody
cenv_name=amr0.4_o8
[ ! -d $cenv_name ] && conda create -y -p ./$cenv_name
echo "source activate ./$cenv_name"
source activate ./$cenv_name

set -o nounset

# for fairseq
conda install python=3.7
conda install pytorch=1.4 -c pytorch

[ ! -d apex ] && git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# install fariseq
[ ! -d fairseq ] && git clone https://github.com/jzhou316/fairseq-stack-transformer.git fairseq
cd fairseq
git checkout v0.3.0/decouple-fairseq
pip install --editable .
cd ..

# install fairseq from the official site
# [ ! -d fairseq ] && git clone https://github.com/pytorch/fairseq
# cd fairseq
# git checkout v0.7.2 
# pip install --editable .
# cd ..

# install transition_amr_parser
pip install spacy ipdb
pip install --editable .

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

