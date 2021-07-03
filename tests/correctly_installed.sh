set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

trap 'case $? in
    139) echo -e "\033[91mCode segfaulted!\033[0m (probably .cuda())\n";;
esac' EXIT

python tests/correctly_installed.py
