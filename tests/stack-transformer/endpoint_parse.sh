set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# sanity check
[ ! -f tests/stack-transformer/endpoint_parse.sh ] && \
    echo "Please call this as bash tests/stack-transformer/endpoint_parse.sh" && \
    exit 1

# QALD
gold_amr=/dccstor/ysuklee1/AMR/treebank/QB20200305/qald_dev2_pass3.jaln
checkpoint=/dccstor/ykt-parse/revanth/analysis/deployed_models/QALD_TACL_Tahira_fix/model.pt
dev_tokenized_sentences=/dccstor/ykt-parse/revanth/oracles/qbqaldlargefinetune_o5+Word100/dev.en

# LDC
#gold_amr=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned

# folder where we write data
rm -f DATA.tests/endpoint.amr DATA.tests/endpoint.smatch
mkdir -p DATA.tests/

# kernprof -l transition_amr_parser/parse.py \
amr-parse \
    --in-tokenized-sentences $dev_tokenized_sentences \
    --in-checkpoint $checkpoint \
    --roberta-batch-size 10 \
    --batch-size 128 \
    --out-amr DATA.tests/endpoint.amr
# python -m line_profiler parse.py.lprof
 
# FIXME: removed for debugging
#    --roberta-cache-path ./cache/roberta.large \

smatch.py \
     --significant 4  \
     -f $gold_amr \
     DATA.tests/endpoint.amr \
     -r 10 \
     > DATA.tests/endpoint.smatch

cat DATA.tests/endpoint.smatch
