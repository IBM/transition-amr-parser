set -o errexit 
set -o pipefail
set -o nounset

[ -d DATA/LP/ ] && rm -R DATA/LP
mkdir -p DATA/LP/corpora/

# Download data

[ ! -f DATA/LP/corpora/dev.txt ] && \
    wget -O DATA/LP/corpora/dev.txt \
        https://amr.isi.edu/download/amr-bank-struct-v1.6-dev.txt

[ ! -f DATA/LP/corpora/train.txt ] && \
    wget -O DATA/LP/corpora/train.txt \
    https://amr.isi.edu/download/amr-bank-struct-v1.6-training.txt 

[ ! -f DATA/LP/corpora/test.txt ] && \
    wget -O DATA/LP/corpora/test.txt \
    wget https://amr.isi.edu/download/amr-bank-struct-v1.6-test.txt
