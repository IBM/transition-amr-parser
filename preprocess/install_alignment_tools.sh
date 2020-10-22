set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset

# work from preprocess
cd preprocess

# JAMR aligner
echo -e "\nDownloading JAMR\n"
[ ! -d jamr ] && git clone https://github.com/jflanigan/jamr.git
cd jamr
git checkout Semeval-2016 
./setup 
. scripts/config.sh 
cd ..

# Kevin aligner
echo -e "\nDownloading Kevin\n"
# FIXME: This URL seems to be no longer active
wget --no-check-certificate https://www.isi.edu/~damghani/papers/Aligner.zip
unzip Aligner.zip 
mv Publish_Version kevin
rm Aligner.zip
cd kevin
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake . 
make 
make install 
cd ..
