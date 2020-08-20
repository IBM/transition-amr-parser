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
pip install --no-deps --editable .

# smatch v1.0.4
[ ! -d smatch.v1.0.4 ] && git clone https://github.com/snowblink14/smatch.git smatch.v1.0.4
cd smatch.v1.0.4
git checkout v1.0.4
cd ..
pip install smatch.v1.0.4/
