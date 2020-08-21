set -o pipefail 
set -o errexit 
. set_environment.sh
set -o nounset
bash tests/state_machine/copy_oracle.sh $LDC2016_AMR_CORPUS/jkaln_2016_scr.txt 0.983
