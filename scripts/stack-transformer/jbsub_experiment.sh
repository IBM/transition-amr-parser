set -o errexit
set -o pipefail

# set host architecture
[ -z "$1" ] && echo -e "\nProvide queue e.g. x86_24h or ppc_6h\n" && exit 1
queue=$1

set -o nounset

# sanity check architecture
if [[ "$queue" =~ ppc_.* ]];then

    # set GPU
    gpu_type=v100

elif [[ "$queue" =~ x86_.* ]];then

    # set GPU
    gpu_type=k80

else    

    # Maybe using the old p9?
    echo -e "\nUnknown queue $queue, must be x86_.*|pcc_.*\n"    
    exit 1


fi

# load config
. scripts/stack-transformer/config.sh

# identify the experiment by the repo tag
repo_tag="$(basename $(pwd) | sed 's@.*\.@@')"

# copy config to own folder
# Loop over seeds
for index in $(seq $num_seeds);do

    # define seed and working dir
    seed=$((41 + $index))
    checkpoints_dir="${checkpoints_dir_root}-seed${seed}/"

    # create repo
    mkdir -p $checkpoints_dir   

    # copy config and store in model folder
    cp scripts/stack-transformer/config.sh $checkpoints_dir/config.sh

done

echo "stage-1: Preprocess"
if [ ! -f "$features_folder/train.en-actions.actions.bin" ];then

    # Remove uncomplete folders
    [ -d $features_folder/ ] && echo "Removing $features_folder" && rm -R $features_folder/

    jbsub_tag="pr-${repo_tag}-$$"
    mkdir -p "$features_folder"

    # run preprocessing
    jbsub -cores "${num_cores}+1" -mem 50g -q "$queue" -require "$gpu_type" \
          -name "$jbsub_tag" \
          -out $features_folder/${jbsub_tag}-%J.stdout \
          -err $features_folder/${jbsub_tag}-%J.stderr \
          /bin/bash scripts/stack-transformer/preprocess.sh $checkpoints_dir/config.sh

    # train will wait for this to start
    train_depends="-depend $jbsub_tag"

else

    # resume from extracted
    train_depends=""

fi

echo "stage-2/3: Training/Testing (multiple seeds)"
# Loop over seeds
for index in $(seq $num_seeds);do

    # define seed and working dir
    seed=$((41 + $index))
    checkpoints_dir="${checkpoints_dir_root}-seed${seed}/"

    if [ ! -f "$checkpoints_dir/checkpoint_best.pt" ];then

        mkdir -p "$checkpoints_dir"

        jbsub_tag="tr-${repo_tag}-s${seed}-$$"

        # run new training
        jbsub -cores 1+1 -mem 50g -q "$queue" -require "$gpu_type" \
              -name "$jbsub_tag" \
              $train_depends \
              -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
              -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
              /bin/bash scripts/stack-transformer/train.sh $checkpoints_dir/config.sh "$seed"

        # testing will wait for this name to start
        test_depends="-depend $jbsub_tag"

    else

        # resume from trained model, start test directly
        test_depends=""

    fi

    # run test on best ce model
    jbsub_tag="dec-${repo_tag}-$$"
    jbsub -cores 1+1 -mem 50g -q "$queue" -require "$gpu_type" \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash scripts/stack-transformer/test.sh $checkpoints_dir/config.sh $checkpoints_dir/$test_basename

    # paralel tester
    jbsub_tag="tdec-${repo_tag}-$$"
    jbsub -cores 1+1 -mem 50g -q "$queue" -require "$gpu_type" \
          -name "$jbsub_tag" \
          $test_depends \
          -out $checkpoints_dir/${jbsub_tag}-%J.stdout \
          -err $checkpoints_dir/${jbsub_tag}-%J.stderr \
          /bin/bash scripts/stack-transformer/epoch_tester.sh $checkpoints_dir/

done
