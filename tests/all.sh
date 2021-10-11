set -o errexit
# all conventional tests
bash tests/correctly_installed.sh
bash tests/minimal_test.sh
bash tests/fairseq_data_iterator.sh
# Smatch computation will take long here
bash tests/oracles/amr_o10.sh DATA/wiki25/aligned/cofill/train.txt
