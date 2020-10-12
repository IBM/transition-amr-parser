set -o errexit
set -o nounset
set -o pipefail

for model in "$@";do

    [ ! -d $model ] && echo "Expected folder $model" && exit 1

    cp $model/config_top3-average_beam10.sh $model/test_config_top3-average_beam10.sh
    sed 's@--gen-subset valid@--gen-subset test@' -i $model/test_config_top3-average_beam10.sh
    echo "$model/test_config_top3-average_beam10.sh" 
    bash scripts/stack-transformer/final_test.sh $model/test_config_top3-average_beam10.sh $model/checkpoint_top3-average.pt
done
