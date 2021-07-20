#
# Check we produce exactly the same ELMO vocabularies for
#
set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

python align_cfg/vocab.py \
    --in-amrs \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/dev.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/test.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/train.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/dev.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/test.txt \
        /dccstor/ykt-parse/SHARED/misc/adrozdov/data/amr2+ontonotes+sQuad.txt \
    --out-folder . 

/dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-unseen-v1
/dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1
/dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/test.txt
/dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.train-v1
