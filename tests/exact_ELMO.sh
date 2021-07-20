#
# Check we produce exactly the same ELMO vocabularies for
#
set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

rm -Rf TMP
mkdir TMP

# /dccstor/ykt-parse/SHARED/misc/adrozdov/data/elmo.01a9bcdf.npy

python align_cfg/vocab.py \
    --in-amrs \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/dev.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/test.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/train.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/dev.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/test.txt \
         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/amr2+ontonotes+sQuad.txt \
    --out-folder TMP

# Extract ELMO embeddings
bash align_cfg/pretrained_embeddings.sh TMP

#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/dev.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/test.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/train.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/dev.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/test.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/amr2+ontonotes+sQuad.txt \

#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-unseen-v1 \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/test.txt \
#         /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.train-v1 \

