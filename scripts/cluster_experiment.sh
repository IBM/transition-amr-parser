set -o errexit
set -o pipefail 
set -o nounset

# sanity checks
[ ! -d scripts ] && echo "to be run as bash scripts/train.sh" && exit 1
[ ! -f "scripts/local_variables_${1}.sh" ] && echo "missing scripts/local_variables_${1}.sh" && exit 1

# load local variables used below
. scripts/local_variables_${1}.sh

# store current hash in model folder
commit_hash=$(git log --pretty=format:'%H' -n 1)
touch models/$name/$commit_hash

# store variables in model folder for traceability and concurrency
cp scripts/local_variables_${1}.sh models/$name/

echo "$cluster_bash scripts/train.sh models/$name/local_variables.sh"
$cluster_bash scripts/train.sh models/$name/local_variables_${1}.sh 
