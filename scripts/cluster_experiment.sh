set -o errexit
set -o pipefail 
set -o nounset

# sanity checks
[ ! -d scripts ] && echo "to be run as bash scripts/train.sh" && exit 1

# load local variables used below
. set_environment.sh

# store current hash in model folder
commit_hash=$(git log --pretty=format:'%H' -n 1)
touch models/$name/$commit_hash

# store variables in model folder for traceability and concurrency
cp set_environment.sh models/$name/

echo "$cluster_bash scripts/train.sh"
$cluster_bash scripts/train.sh
