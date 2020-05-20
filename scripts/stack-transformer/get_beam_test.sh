set -o errexit
set -o pipefail
# environment variables
. set_environment.sh
# argument handling
script_help="$0 <beam size> <checkpoint tag> <checkpoint 1> [<checkpoint 2>]"
[ -z $1 ] && echo "$script_help" && exit 1
[ -z $2 ] && echo "$script_help" && exit 1
[ -z $3 ] && echo "$script_help" && exit 1
[ -f $2 ] && echo "$script_help" && exit 1
set -o nounset

# Take first argument as beam size and pop it
beam_size=$1
tag=$2

for checkpoint in ${@:3};do
    
    [ ! -f $checkpoint ] && \
        echo "Missing checkpoint: $checkpoint" && \
        exit 1

    # Skip if no config
    checkpoints_folder=$(dirname $checkpoint)
    config="$checkpoints_folder/config.sh"
    [ ! -f $config ] && continue
    
    # create config
    if [ "$tag" == "" ];then
        TEST_TAG=beam${beam_size}
    else
        TEST_TAG=${tag}_beam${beam_size}
    fi

    beam_config=$(dirname $config)/config_${TEST_TAG}.sh
    echo "cp $config $beam_config"
    cp $config $beam_config
    sed "s@beam_size=.*@beam_size=$beam_size@" -i $beam_config
    sed "s@TEST_TAG=.*@TEST_TAG=$TEST_TAG@" -i $beam_config
    echo "Created $beam_config"

    # run test
    bash scripts/stack-transformer/test.sh $beam_config $checkpoint
    
done
