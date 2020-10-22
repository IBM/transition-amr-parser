set -o errexit
set -o pipefail
[ "$#" -lt 2 ] && echo "$0 LDC_CORPUS CORPUS_FOLDER" && exit 1
LDC_FOLDER=$1
CORPUS_FOLDER=$2
. set_environment.sh
set -o nounset

# Were normalized/aligned data will be stored
mkdir -p $CORPUS_FOLDER

for sset in dev train test;do

    if [ "$sset" == "train" ];then
        folder=$LDC_FOLDER/training/
    else
        folder=$LDC_FOLDER/$sset/
    fi    

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
