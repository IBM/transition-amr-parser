# Removes all checkpoints that are not _last _best and_best_SMATCH from a given
# checkpoints folder
#
# ATTENTION: at this point this includes _second_best_SMATCH and
# third_best_SMATCH used for weight ensembling. This also eliminates the
# possibility of other weight ensembling strategies
#
set -o errexit
set -o pipefail 
[ -z $1 ] && echo -e "\n$0 <folder with checkpoints>\n" && exit 1
model_dir=$1
set -o nounset

# ensure its no empty relative path
[ "$model_dir" == "" ] && \
    echo -e "\nfolder can not be empty, exiting\n" && \
    exit 1 
model_dir=$(realpath $model_dir)

# assume we are soliciting best 40 models
# TODO: Have it as variable and check for it in the config
[ $(find $model_dir/epoch_tests/ -iname '*.actions' | wc -l) != "40" ] && \
    echo -e "\nmust have results for 40 models \n" && \
    exit 1 

# exit if there are no "*_best_*.pt" soft-links e.g. _best_SMATCH.pt
best_score_links=$(find "$model_dir" -iname '*_best_*.pt')
[ "$best_score_links" == "" ] && \
    echo -e "\nmust have *_best_*.pt checkpoints, exiting \n" && \
    exit 1 

# check for soft-links to checkpoints and replace them by copies of the
# checkpoints
for best_link in $(ls $model_dir/*_best*.pt);do

    # skip if its not a soft link
    [ ! -L "$best_link" ] && continue
    
    # get checkpoint pointed by best smatch. Relink using relative softlink and
    # ensure that exists
    best_checkpoint_original=$(readlink $best_link)
    basename=$(basename $best_checkpoint_original)
    best_checkpoint="$model_dir/$basename"
    [ ! -f "$best_checkpoint" ] && \
        echo -e "\nMissing $best_checkpoint\n" && \
        exit 1

    echo "Saving $(basename $best_checkpoint) ($(basename $best_link))"
    rm $best_link
    mv $best_checkpoint $best_link
    
done

# remove old link replace it by checkpoint
echo "Removing checkpoint[0-9]*.pt"
# remove all other checkpoints
for dfile in $(find $model_dir -type f -iname 'checkpoint[0-9]*.pt' | sort -n);do
    rm $dfile
done
