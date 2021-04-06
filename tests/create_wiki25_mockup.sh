set -o errexit 
set -o pipefail
set -o nounset 

# This simulates conventional corpora using the 25 wiki sentences
rm -Rf DATA/wiki25/
# Create original data
mkdir -p DATA/wiki25/corpora/
ln -s ../../wiki25.jkaln DATA/wiki25/corpora/train.txt 
ln -s ../../wiki25.jkaln DATA/wiki25/corpora/dev.txt
ln -s ../../wiki25.jkaln DATA/wiki25/corpora/test.txt 
touch DATA/wiki25/corpora/.done

# Simulate aligned data from wiki25
mkdir -p DATA/wiki25/aligned/cofill/
ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/train.txt 
ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/dev.txt
ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/test.txt 
touch DATA/wiki25/aligned/cofill/.done
