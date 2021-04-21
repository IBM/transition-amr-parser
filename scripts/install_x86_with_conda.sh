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

### local env
# cenv_name=amr0.4_ody
# cenv_name=amr0.4_bart-o10
# [ ! -d $cenv_name ] && conda create -y -p ./$cenv_name
# echo "source activate ./$cenv_name"
# source activate ./$cenv_name

# for fairseq
conda install python=3.7 -y
conda install pytorch=1.4 -c pytorch -y
# conda install pytorch==1.4 cudatoolkit=10.0 -c pytorch -y

# install apex
[ -d apex ] && rm -rf apex
[ ! -d apex ] && git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# ## from fairseq installation command: not working on my machine
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#   --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#   --global-option="--fast_multihead_attn" ./
cd ..

# below doesn't work without the installation flags
# conda install -c conda-forge nvidia-apex -y

# install fariseq
# [ ! -d fairseq ] && git clone https://github.com/jzhou316/fairseq-stack-transformer.git fairseq
# cd fairseq
# git checkout v0.3.0/decouple-fairseq
# pip install --editable .
# cd ..

# install fairseq from the official site
# [ ! -d fairseq ] && git clone https://github.com/pytorch/fairseq
# cd fairseq
# git checkout v0.7.2
# pip install --editable .
# cd ..

pip install fairseq==0.10.2

# install tensorboardX
pip install tensorboardX
# install packaging for compatibility of PyTorch < 1.2
conda install packaging -y
pip install penman

# install transition_amr_parser
pip install spacy ipdb
pip install --editable .

# initialize the working environment
mkdir ./jbsub_logs

# install pytorch_scatter
[ -d pytorch_scatter ] && rm -rf pytorch_scatter
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
git checkout 1.3.2
# ## not sure whether below is needed for ccc
# Ensure modern GCC
# export GCC_DIR=/opt/share/gcc-5.4.0/x86_64/
# export PATH=/opt/share/cuda-9.0/x86_64/bin:$GCC_DIR/bin:$PATH
# export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
# ##
python setup.py develop
cd ..

# install smatch v1.0.4
[ -d smatch ] && rm -rf smatch
[ ! -d smatch ] && git clone https://github.com/snowblink14/smatch.git smatch
cd smatch
git checkout v1.0.4
cd ..
pip install smatch/
## if the above doesn't work with due to an EnvironmentError: [Errno 13]
# python setup.py install
# cd ..

### to enable jupyter notebook environment
# conda install jupyter
# conda install nb_conda

