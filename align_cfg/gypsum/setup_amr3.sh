#!/bin/bash

TASK="AMR3.0"
CACHE="cache-amr3"

mkdir -p $CACHE

cp ./DATA/${TASK}/aligned/cofill/train.txt ./${CACHE}/train.aligned.txt

python preprocess/remove_wiki.py ./DATA/${TASK}/corpora/dev.txt ./DATA/${TASK}/corpora/dev.txt.no_wiki
python preprocess/remove_wiki.py ./DATA/${TASK}/corpora/test.txt ./DATA/${TASK}/corpora/test.txt.no_wiki
python preprocess/remove_wiki.py ./DATA/${TASK}/corpora/train.txt ./DATA/${TASK}/corpora/train.txt.no_wiki

python align_cfg/tokenize_amr.py --in-amr ./DATA/${TASK}/corpora/dev.txt.no_wiki --out-amr ./${cache}/dev.txt.no_wiki
python align_cfg/tokenize_amr.py --in-amr ./DATA/${TASK}/corpora/test.txt.no_wiki --out-amr ./${cache}/test.txt.no_wiki
python align_cfg/tokenize_amr.py --in-amr ./DATA/${TASK}/corpora/train.txt.no_wiki --out-amr ./${cache}/train.txt.no_wiki

python align_cfg/vocab.py \
    --in-amrs \
        ./DATA/${TASK}/aligned/cofill/dev.txt \
        ./DATA/${TASK}/aligned/cofill/test.txt \
        ./DATA/${TASK}/aligned/cofill/train.txt \
        \
        ./DATA/${TASK}/corpora/dev.txt \
        ./DATA/${TASK}/corpora/test.txt \
        ./DATA/${TASK}/corpora/train.txt \
        \
        ./DATA/${TASK}/corpora/dev.txt.no_wiki \
        ./DATA/${TASK}/corpora/test.txt.no_wiki \
        ./DATA/${TASK}/corpora/train.txt.no_wiki \
    --out-text ./${CACHE}/vocab.text.txt \
    --out-amr ./${CACHE}/vocab.amr.txt

python align_cfg/pretrained_embeddings.py --cuda --cache-dir ./${CACHE}/ --vocab ./${CACHE}/vocab.text.txt
python align_cfg/pretrained_embeddings.py --cuda --cache-dir ./${CACHE}/ --vocab ./${CACHE}/vocab.amr.txt

cp align_cfg/setup_amr3.sh $CACHE/setup_data.sh

