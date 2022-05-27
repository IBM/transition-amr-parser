set -o errexit
# all conventional tests
bash tests/correctly_installed.sh
# small test with 25 sentences
bash tests/minimal_test.sh
# standalone parser
bash tests/standalone.sh
# oracle for wiki25 imperfect due to alignments
bash tests/oracles/amr_o10.sh DATA/wiki25/aligned/cofill/train.txt

# we need to reach here
printf "[\033[92m OK \033[0m] $0\n"
