set -o errexit 
set -o pipefail
. set_environment.sh
set -o nounset 

python run/status.py $@
