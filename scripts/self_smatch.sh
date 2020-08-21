set -o errexit 
set -o pipefail
#. set_environment.sh
[ -z $1 ] && echo "$0 <amr file>" && exit 1
amr_file=$1
set -o nounset 
python smatch/smatch.py --significant 4 -r 10 --ms -f $amr_file $amr_file
