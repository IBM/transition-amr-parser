set -o errexit
set -o pipefail 
set -o nounset

# this assumes python 3.6+ available

# modules
pip install virtualenv --upgrade
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt

# spacy lemmatization
python -m spacy download en

# smatch
git clone https://github.com/snowblink14/smatch.git

# detailed smatch (wil need python2)
# git clone https://github.com/mdtux89/amr-evaluation
# cd amr-evaluation
# pyenv local pypy2.7-7.0.0
