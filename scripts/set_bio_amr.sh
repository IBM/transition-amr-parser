set -o nounset
set -o pipefail
set -o errexit

CORPUS_FOLDER=DATA/AMR/corpora/bio_amr2.0/
mkdir -p $CORPUS_FOLDER
cd $CORPUS_FOLDER
[ ! -f train.txt ] && \
    wget --no-check-certificate https://amr.isi.edu/download/2016-03-14/amr-release-training-bio.txt && \
    mv amr-release-training-bio.txt train.txt && \
    sed '1,2d' -i train.txt
[ ! -f dev.txt ] && \
    wget --no-check-certificate https://amr.isi.edu/download/2016-03-14/amr-release-dev-bio.txt && \
    mv amr-release-dev-bio.txt dev.txt && \
    sed '1,2d' -i dev.txt
[ ! -f test.txt ] && \
    wget --no-check-certificate https://amr.isi.edu/download/2016-03-14/amr-release-test-bio.txt && \
    mv amr-release-test-bio.txt test.txt && \
    sed '1,2d' -i test.txt
cd -
