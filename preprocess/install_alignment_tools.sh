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
# fix outdated perl expression
sed 's@\(.*tamil\)@#\1@' -i tools/cdec/corpus/support/quote-norm.pl
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
