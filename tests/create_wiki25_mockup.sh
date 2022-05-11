set -o errexit 
set -o pipefail
set -o nounset 

# This simulates conventional corpora using the 25 wiki sentences
# Create original data
mkdir -p DATA/wiki25/corpora/
cp DATA/wiki25.jkaln DATA/wiki25/corpora/train.txt 
# remove JAMR meta-data, which we do not have in reality
sed -i.bak '/^# ::tok.*/d' DATA/wiki25/corpora/train.txt
sed -i.bak '/^# ::node.*/d' DATA/wiki25/corpora/train.txt
sed -i.bak '/^# ::edge.*/d' DATA/wiki25/corpora/train.txt
sed -i.bak '/^# ::root.*/d' DATA/wiki25/corpora/train.txt
sed -i.bak '/^# ::alignments.*/d' DATA/wiki25/corpora/train.txt
[ ! -f DATA/wiki25/corpora/dev.txt ] \
    && ln -s ./train.txt DATA/wiki25/corpora/dev.txt
[ ! -f DATA/wiki25/corpora/test.txt ] \
    && ln -s ./train.txt DATA/wiki25/corpora/test.txt 

touch DATA/wiki25/corpora/.done

# Simulate aligned data from wiki25
mkdir -p DATA/wiki25/aligned/cofill/
[ ! -f DATA/wiki25/aligned/cofill/train.txt ] \
    && ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/train.txt 
[ ! -f DATA/wiki25/aligned/cofill/dev.txt ] \
    && ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/dev.txt
[ ! -f DATA/wiki25/aligned/cofill/test.txt ] \
    && ln -s ../../../wiki25.jkaln DATA/wiki25/aligned/cofill/test.txt 
touch DATA/wiki25/aligned/cofill/.done
