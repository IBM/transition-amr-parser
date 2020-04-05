set -o errexit 
set -o pipefail
set -o nounset 

# oracle tests
bash tests/state_machine/o3+W.sh dev
# this can take 30-40min due to smatch
# bash tests/state_machine/o3+W.sh train  

# stack-LSTM tests
bash tests/stack-LSTM/decode.sh

# stack-Transformer tests
# bash tests/stack-transformer/decode.sh
