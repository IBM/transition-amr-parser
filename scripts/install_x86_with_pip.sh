set -o errexit
set -o pipefail 
# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh
set -o nounset

# fairseq
[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
pip install -r scripts/stack-transformer/requirements.txt
cd fairseq
git checkout v0.3.0/decouple-fairseq
pip install --editable .
cd ..

# this repo without the dependencies (included in fairseq)
cp setup.py _setup.py.saved
sed '/install_requires=install_requires,/d' -i setup.py
pip install --editable .
mv _setup.py.saved setup.py

# smatch
[ ! -d smatch ] && git clone git@github.ibm.com:mnlp/smatch.git
cd smatch
git checkout f728c3d3f4a71b44678224d6934c1e67c4d37b89
cd ..
pip install smatch/
