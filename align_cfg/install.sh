set -o errexit
set -o pipefail

# activate normal env
. set_environment.sh

# load normal env
# install DGL, in addition to normall install in README
conda install -y -c dglteam "dgl-cuda10.1<0.5"

# install separate env for ELMO
# exit current virtualenv
conda deactivate
# create and install in new env
[ ! -d cenv_ELMO ] && conda create -y -p ./cenv_ELMO
conda activate ./cenv_ELMO
conda install -y pip
pip install allennlp
conda deactivate 
