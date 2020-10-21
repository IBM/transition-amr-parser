set -o nounset
set -o errexit
set -o pipefail

# patch fairseq
if [ ! -d fairseq-stack-transformer-v0.3.2 ];then
    git clone https://github.com/pytorch/fairseq.git fairseq-stack-transformer-v0.3.2
    cd fairseq-stack-transformer-v0.3.2
    git checkout -b stack-transformer-v0.3.2 a33ac06 
    git apply ../transition_amr_parser/stack_transformer/fairseq_a33ac06.patch
    cd ..
fi
