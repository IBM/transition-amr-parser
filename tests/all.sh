set -o errexit
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

# all conventional tests
bash tests/correctly_installed.sh
# small test with 25 sentences
bash tests/minimal_test.sh
# standalone parser
bash tests/standalone.sh
# oracle for wiki25 imperfect due to alignments
bash tests/oracles/amr_o10.sh DATA/wiki25/aligned/cofill_isi/train.txt
# if we reach here, we are good
TESTS_PASSED="Y"
