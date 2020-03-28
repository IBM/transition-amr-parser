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
[ "$model_dir" == "" ] && echo -e "\nno empty folders\n" && exit 1 
model_dir=$(realpath $model_dir)

# locate checkpoint best
best_link="$model_dir/checkpoint_best_SMATCH.pt"
# exit if it has no checkpoint best 
[ ! -L "$best_link" ] && \
    echo -e "\nMissing $best_link\n" && \
    exit 1

# get checkpoint pointed by best smatch. Relink using relative softlink and
# ensure that exists
best_checkpoint_original=$(readlink $best_link)
basename=$(basename $best_checkpoint_original)
best_checkpoint="$model_dir/$basename"
[ ! -f "$best_checkpoint" ] && \
    echo -e "\nMissing $best_checkpoint\n" && \
    exit 1

# remove old link replace it by checkpoint
echo ""
echo "rm $best_link"
echo "mv $best_checkpoint $best_link"
# remove all other checkpoints
for dfile in $(find $model_dir -type f -iname 'checkpoint[0-9]*.pt' | sort -n);do
    echo "rm $dfile"
done
# leave a softlink to best checkpoint
echo "ln -s $best_link $best_checkpoint"
echo ""
echo "Following actions will be performed, see above"
read -p "Are you sure you want to proceed? y/n" answer
if [ "$answer" == "y" ];then
    rm $best_link
    mv $best_checkpoint $best_link
    find $model_dir -type f -iname 'checkpoint[0-9]*.pt' -delete
    ln -s $best_link $best_checkpoint
else
    echo "aborting"
fi
