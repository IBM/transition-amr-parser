set -o pipefail 
set -o errexit 
. set_environment.sh
set -o nounset
bash tests/state_machine/copy_oracle.sh $LDC2017_AMR_CORPUS/dev.txt 0.953
