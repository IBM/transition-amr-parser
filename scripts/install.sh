set -o errexit
set -o pipefail 
set -o nounset

pip install --editable .

# Install fairseq
[ ! -d fairseq ] && git clone git@github.ibm.com:ramon-astudillo/fairseq.git
pip install -r fairseq/requirements.txt
pip install --editable fairseq/

# Make smatch importable and editable, use MNLPs smatch (importable and faster) 
rm -Rf smatch
git clone git@github.ibm.com:mnlp/smatch.git
# editable just in case
pip install --editable smatch/
