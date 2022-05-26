#
# Check we produce exactly the correct vocab files.
#
set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

rm -Rf TMP
mkdir TMP

DATA_DIR="/dccstor/ykt-parse/SHARED/misc/adrozdov/data"

# Pre-computed embeddings should have hashes:
# - text: /dccstor/ykt-parse/SHARED/misc/adrozdov/data/elmo.e257682c.npy
# - amr:  /dccstor/ykt-parse/SHARED/misc/adrozdov/data/elmo.03e30112.npy

python ibm_neural_aligner/vocab.py \
    --in-amrs \
         ${DATA_DIR}/AMR2.0/aligned/cofill/train.txt \
         ${DATA_DIR}/AMR2.0/aligned/cofill/dev.txt \
         ${DATA_DIR}/AMR2.0/aligned/cofill/test.txt \
         ${DATA_DIR}/AMR3.0/aligned/cofill/train.txt \
         ${DATA_DIR}/AMR3.0/aligned/cofill/dev.txt \
         ${DATA_DIR}/AMR3.0/aligned/cofill/test.txt \
         ${DATA_DIR}/amr2+ontonotes+squad.txt \
    --out-text TMP/vocab.text.txt \
    --out-amr TMP/vocab.amr.txt

# Extract ELMO embeddings
bash ibm_neural_aligner/pretrained_embeddings.sh TMP
