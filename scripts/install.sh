set -o errexit
set -o pipefail 
set -o nounset

pip install --editable .

# Install fairseq
[ -d fairseq ] && echo "fairseq/ exists, remove to reinstall" && exit 1
git clone git@github.ibm.com:ramon-astudillo/fairseq.git
cd fairseq
pip install --editable .

# Make smatch importable and editable, use MNLPs smatch (importable and faster) 
rm -Rf smatch
git clone git@github.ibm.com:mnlp/smatch.git
# editable just in case
pip install --editable smatch/
