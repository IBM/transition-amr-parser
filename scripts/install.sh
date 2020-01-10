set -o errexit
set -o pipefail 
set -o nounset

# See README for instructions on how to define this. You can comment this if
# you are ok with instal on your active python version
. set_environment.sh

# Install requirements nad main module by separate. This is needed for
# compatibility with PPC installer
pip install -r requirements.txt
pip install --editable .
