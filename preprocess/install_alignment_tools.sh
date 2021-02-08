set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset

# work from preprocess
cd preprocess

# JAMR aligner
echo -e "\nDownloading JAMR\n"
rm -Rf jamr
git clone https://github.com/jflanigan/jamr.git
cd jamr
git checkout Semeval-2016 
./setup 
. scripts/config.sh 
cd ..

# Kevin aligner
echo -e "\nDownloading Kevin\n"
git clone https://github.com/damghani/AMR_Aligner
mv AMR_Aligner kevin
cd kevin
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake . 
make 
make install 
cd ..
