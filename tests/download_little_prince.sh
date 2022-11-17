set -o errexit 
set -o pipefail
set -o nounset

# [ -d DATA/LP/ ] && rm -R DATA/LP
mkdir -p DATA/LP/corpora/

# Download data

if [ ! -f DATA/LP/corpora/dev.txt ];then
    wget --no-check-certificate -O DATA/LP/corpora/dev.txt.tmp https://amr.isi.edu/download/amr-bank-struct-v1.6-dev.txt 
    sed '1,2d' DATA/LP/corpora/dev.txt.tmp > DATA/LP/corpora/dev.txt    
    rm DATA/LP/corpora/dev.txt.tmp
fi

if [ ! -f DATA/LP/corpora/train.txt ];then
    wget --no-check-certificate -O DATA/LP/corpora/train.txt.tmp https://amr.isi.edu/download/amr-bank-struct-v1.6-training.txt 
    sed '1,2d' DATA/LP/corpora/train.txt.tmp > DATA/LP/corpora/train.txt
    rm DATA/LP/corpora/train.txt.tmp
fi    

if [ ! -f DATA/LP/corpora/test.txt ];then
    wget --no-check-certificate -O DATA/LP/corpora/test.txt.tmp https://amr.isi.edu/download/amr-bank-struct-v1.6-test.txt 
    sed '1,2d' DATA/LP/corpora/test.txt.tmp > DATA/LP/corpora/test.txt 
    rm DATA/LP/corpora/test.txt.tmp
fi
