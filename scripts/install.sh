set -o errexit
set -o pipefail 
set -o nounset

pip install --editable .

# spacy lemmatization
python -m spacy download en

# Make smatch importable and editable, use MNLPs smatch (importable and faster) 
rm -Rf smatch
git clone git@github.ibm.com:mnlp/smatch.git
# editable just in case
pip install --editable smatch/

# detailed smatch (wil need python2)
# git clone https://github.com/mdtux89/amr-evaluation
# cd amr-evaluation
# pyenv local pypy2.7-7.0.0
