set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
set -o nounset

# Argument handling
config=$1

. $config

model_folder=$(dirname $config)

for amr_file in $(find $model_folder -iname '*.amr' | grep -v 'wiki.amr');do

    # Get name of targte amr file
    amr_basename=$(basename $amr_file .amr)
    wiki_amr_file=$(dirname $amr_file)/${amr_basename}.wiki.amr
    wiki_smatch_file=$(dirname $amr_file)/${amr_basename}.wiki.smatch

    # Add AMR wiki 
    python fairseq/dcc/add_wiki.py $amr_file $WIKI_DEV > $wiki_amr_file 

    # Compute smatch
    smatch.py \
         --significant 4  \
         -f $AMR_DEV_FILE_WIKI \
         $wiki_amr_file \
         -r 10 \
         > $wiki_smatch_file

    printf "\r$wiki_smatch_file "
    cat $wiki_smatch_file

done
