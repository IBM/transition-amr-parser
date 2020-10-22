set -o errexit
set -o pipefail
[ -z "$1" ] && echo "$0 LDC_CORPUS" && exit 1
. set_environment.sh
set -o nounset

# Were normalized/aligned data will be stored
mkdir -p $CORPUS_FOLDER

for sset in dev train test;do

    if [ "$sset" == "train" ];do
        folder=$LDC_FOLDER/training/
    else
        folder=$LDC_FOLDER/$sset/

    # Merge domains into single files and normalize
    python preprocess/merge_files.py \
        $folder \
        $CORPUS_FOLDER/${sset}.txt
    
    # Remove wiki
    python preprocess/remove_wiki.py \
        $CORPUS_FOLDER/${sset}.txt \
        $CORPUS_FOLDER/${sset}.no_wiki.txt
    
    # align
    bash preprocess/align.sh \
        $CORPUS_FOLDER/${sset}.no_wiki.txt \
        $CORPUS_FOLDER/${sset}.no_wiki.aligned.txt

done
