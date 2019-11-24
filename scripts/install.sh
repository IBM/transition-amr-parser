set -o errexit
set -o pipefail 
# Allow for extenal virtualenv for install as dependent module
if [ -z "$1" ];then
    virtualenv_name=venv
else
    virtualenv_name=$1
fi
set -o nounset

# this assumes python 3.6+ available

# modules
if [ ! -d $virtualenv_name ];then
    pip install virtualenv --upgrade
    virtualenv  $virtualenv_name
    .  ${virtualenv_name}/bin/activate
    pip install --editable .
else
    echo "Will use existing $virtualenv_name, removet to force re-install"
fi

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
