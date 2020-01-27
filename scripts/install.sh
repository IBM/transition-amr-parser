set -o errexit
set -o pipefail 
set -o nounset

# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh

# fairseq
[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
pip install -r fairseq/requirements.txt
pip install --editable fairseq/

# this repo without the dependencies (included in fairseq)
sed '/install_requires=install_requires,/d' -i setup.py
pip install --editable .

# smatch
[ ! -d smatch ] && git clone git@github.ibm.com:mnlp/smatch.git
cd smatch
git checkout f728c3d3f4a71b44678224d6934c1e67c4d37b89
cd ..
pip install smatch/
