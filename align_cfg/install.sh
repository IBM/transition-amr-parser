set -o errexit
set -o pipefail

# activate normal env
. set_environment.sh

# load normal env
# install DGL, in addition to normall install in README
conda install -y -c dglteam "dgl-cuda10.1<0.5"
