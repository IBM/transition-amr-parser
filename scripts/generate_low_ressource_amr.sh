set -o pipefail
set -o errexit
#. set_environment.sh
set -o nounset

#
# Create Multi-Task Dataset with simulated low ressource from AMR2.0
#

OUT_FOLDER=/dccstor/ykt-parse/SHARED/CORPORA/AMR/low_ressource/amr2.0_from_amr1.0_v0/

# Split AMR2.0 with o3 alignments into a set with AMR1.0 ids and one without
mkdir -p amr2.0_from_amr1.0/ amr2.0_minus_amr1.0/
python scripts/amr2.0_from_amr1.0.py \
    /dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/jkaln_2016_scr.txt \
    /dccstor/multi-parse/transformer-amr/AMR_1.0/AMR_1.0_train_jkaln.txt \
    $OUT_FOLDER/amr2.0_from_amr1.0.amr \
    $OUT_FOLDER/amr2.0_minus_amr1.0.amr

# Create oracles (`o3`) without labeled SHIFT for the 25K sentences outside of
# AMR1.0
amr-oracle \
    --in-amr $OUT_FOLDER/amr2.0_minus_amr1.0.amr \
    --out-sentences $OUT_FOLDER/train.en \
    --out-actions $OUT_FOLDER/train.actions \
    --out-rule-stats $OUT_FOLDER/train.rules.json \
    --copy-lemma-action

# Create BIO tags from alignments and node actions of that oracle (PRED, ADDNODE)
amr-fake-parse \
    --in-sentences $OUT_FOLDER/train.en \
    --in-actions $OUT_FOLDER/train.actions \
    --out-bio-tags $OUT_FOLDER/amr2.0_minus_amr1.0.tags

# Remove oracle
rm -f $OUT_FOLDER/train.en $OUT_FOLDER/train.actions $OUT_FOLDER/train.rules.json

# Filter PRED, ADDNODE to generate Word Sense Disambiguation WSD data. Rename
# ADDNODE as Macro (MCR). This includes NER but also othe constructus such as
# reification
python scripts/tasks_from_amr_tags.py \
    $OUT_FOLDER/amr2.0_minus_amr1.0.tags \
    $OUT_FOLDER/amr2.0_minus_amr1.0
