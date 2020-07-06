set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset
python tests/correctly_installed.py
