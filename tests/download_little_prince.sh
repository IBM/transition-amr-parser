set -o errexit 
set -o pipefail
set -o nounset

[ -d DATA/LP/ ] && rm -R DATA/LP
mkdir -p DATA/LP/corpora/

# Download data

if [ ! -f DATA/LP/corpora/dev.txt ];then
    wget -O DATA/LP/corpora/dev.txt https://amr.isi.edu/download/amr-bank-struct-v1.6-dev.txt 
    sed '1,2d' -i DATA/LP/corpora/dev.txt    
fi

if [ ! -f DATA/LP/corpora/train.txt ];then
    wget -O DATA/LP/corpora/train.txt https://amr.isi.edu/download/amr-bank-struct-v1.6-training.txt 
    sed '1,2d' -i DATA/LP/corpora/train.txt    
fi    

if [ ! -f DATA/LP/corpora/test.txt ];then
    wget -O DATA/LP/corpora/test.txt https://amr.isi.edu/download/amr-bank-struct-v1.6-test.txt 
    sed '1,2d' -i DATA/LP/corpora/test.txt    
fi
