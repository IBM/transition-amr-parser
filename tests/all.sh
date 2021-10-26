set -o errexit
# all conventional tests
bash tests/correctly_installed.sh
bash tests/minimal_test.sh
bash tests/fairseq_data_iterator.sh \
    DATA/wiki25/features/cofill_o10_act-states_bart.base/ \
    DATA/wiki25/embeddings/bart.base/
# Smatch computation will take long here
bash tests/oracles/amr_o10.sh DATA/wiki25/aligned/cofill/train.txt
