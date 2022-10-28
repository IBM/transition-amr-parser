set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset
# This will ensure that early exit shows tests having failed
function check_tests_passed {
        if [ "$TESTS_PASSED" == "Y" ];then
                printf "[\033[92m OK \033[0m] $0\n"
        else
                printf "[\033[91m FAILED \033[0m] $0\n"
        fi
}
trap check_tests_passed EXIT
TESTS_PASSED="N"

# python tests/align_mode.py DATA/wiki25/aligned/cofill_isi/train.txt
python tests/align_mode.py DATA/AMR2.0/aligned/cofill/train.txt

# if we reach here, we are good
TESTS_PASSED="Y"
