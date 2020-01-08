set -o errexit 
set -o pipefail
set -o nounset 

# Load config
. $1/config.sh

# 
echo "Linking valid in place of test"
for test_file in $(ls $features_folder/test* | grep -v '\.saved$');do
    mv $test_file ${test_file}.saved
    dev_file=$(echo $test_file | sed 's@test@valid@')
    ln -s $(realpath $dev_file) $test_file
done
