set -o errexit 
set -o pipefail
[ "$#" -lt 2 ] && echo "$0 amr_file aligned_amr_file"  && exit 1
amr_file=$1
aligned_amr_file=$2
. set_environment.sh
set -o nounset

# Use absolute path to avoid problems
amr_file=$(realpath $amr_file)

echo -e "\nRunning JAMR\n"
. preprocess/jamr/scripts/config.sh 
bash preprocess/jamr/scripts/ALIGN.sh < $amr_file > ${amr_file}.jamr

echo -e "\nBuild Kevin input files\n"
# Will create by replacing jamr
# ${amr_file}.amrs 
# ${amr_file}.sents
# ${amr_file}.bad_amrs
python preprocess/jamr_2_kevin.py ${amr_file}.jamr 

echo -e "\nRunning Kevin\n"
bash preprocess/run_kevin_aligner.sh \
    ${amr_file}.amrs \
    ${amr_file}.sents \
    ${amr_file}.kevin

echo -e "\nClean Kevin output\n"
python preprocess/clean_kevin.py ${amr_file}.kevin ${amr_file}.bad_amrs

echo -e "\nMerge Alignments\n"
cd preprocess/merge_scripts
bash run_jtok.sh $(realpath ${amr_file}.kevin) $(realpath ${amr_file}.jamr)
mv $(realpath ${amr_file}.kevin).mrged $aligned_amr_file
rm 1 2 3 4 5 6 7 8 9
cd ../..

echo -e "\nClean\n"
python preprocess/clean.py $aligned_amr_file
