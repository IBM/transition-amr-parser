set -o nounset
set -o errexit
set -o pipefail

# patch fairseq
if [ ! -d fairseq-stack-transformer ];then
    git clone https://github.com/pytorch/fairseq.git fairseq-stack-transformer
    cd fairseq-stack-transformer
    git checkout -b stack-transformer a33ac06 
    git apply ../transition_amr_parser/stack_transformer/fairseq_a33ac06.patch
    cd ..
fi
