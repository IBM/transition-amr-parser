set -o pipefail
set -o errexit
. set_environment.sh
set -o nounset

#
# Create Multi-Task Dataset with simulated low ressource from AMR2.0
#

# TRAIN DATA

OUT_FOLDER_ROOT=/dccstor/ykt-parse/SHARED/CORPORA/AMR/low_ressource/

for max_sentences in amr1.0 5000 2500;do

    # Folder for the corpus
    OUT_FOLDER=$OUT_FOLDER_ROOT/amr2.0_from_${max_sentences}/
    rm -Rf $OUT_FOLDER
    mkdir -p $OUT_FOLDER 

    # Extract entity rules __for the entire set__
    python scripts/extract_rules.py \
        $LDC2016_AMR_CORPUS/jkaln_2016_scr.txt \
        $OUT_FOLDER/jkaln_2016_scr.entity_rules.json

    # Split AMR2.0 with o3 alignments into a set with AMR1.0 ids and one without
    if [ $max_sentences == "amr1.0" ];then
        python scripts/amr2.0_from_amr1.0.py \
            --in-amr $LDC2016_AMR_CORPUS/jkaln_2016_scr.txt \
            --in-amr1 /dccstor/multi-parse/transformer-amr/AMR_1.0/AMR_1.0_train_jkaln.txt \
            --out-amr-from-amr1 $OUT_FOLDER/amr2.0_from_${max_sentences}_train_jkaln.amr \
            --out-amr-reminder $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln.amr
    else
        python scripts/amr2.0_from_amr1.0.py \
            --in-amr $LDC2016_AMR_CORPUS/jkaln_2016_scr.txt \
            --in-amr1 /dccstor/multi-parse/transformer-amr/AMR_1.0/AMR_1.0_train_jkaln.txt \
            --out-amr-from-amr1 $OUT_FOLDER/amr2.0_from_${max_sentences}_train_jkaln.amr \
            --out-amr-reminder $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln.amr \
            --max-sentences $max_sentences
    fi
   
    # Create oracles (`o3`) without labeled SHIFT for the 25K sentences outside of
    # AMR1.0
    amr-oracle \
        --in-amr $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln.amr \
        --entity-rules $OUT_FOLDER/jkaln_2016_scr.entity_rules.json \
        --out-sentences $OUT_FOLDER/train.en \
        --out-actions $OUT_FOLDER/train.actions \
        --out-rule-stats $OUT_FOLDER/train.rules.json \
        --copy-lemma-action
    
    # Create BIO tags from alignments and node actions of that oracle (PRED, ADDNODE)
    amr-fake-parse \
        --in-sentences $OUT_FOLDER/train.en \
        --in-actions $OUT_FOLDER/train.actions \
        --entity-rules $OUT_FOLDER/jkaln_2016_scr.entity_rules.json \
        --out-bio-tags $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln.tags
    
    # Remove oracle
    rm -f $OUT_FOLDER/train.en $OUT_FOLDER/train.actions $OUT_FOLDER/train.rules.json
    
    # Filter PRED, ADDNODE to generate Word Sense Disambiguation WSD data. Rename
    # ADDNODE as Macro (MCR). This includes NER but also othe constructus such as
    # reification
    python scripts/tasks_from_amr_tags.py \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln.tags \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_jkaln
    
    # DEV/TEST DATA
    for sset in dev test;do
        
        amr-oracle \
            --in-amr $LDC2016_AMR_CORPUS/${sset}.txt.removedWiki.noempty.JAMRaligned \
            --entity-rules $OUT_FOLDER/jkaln_2016_scr.entity_rules.json \
            --out-sentences $OUT_FOLDER/${sset}.en \
            --out-actions $OUT_FOLDER/${sset}.actions \
            --copy-lemma-action
        
        # Create BIO tags from alignments and node actions of that oracle (PRED,
        # ADDNODE)
        amr-fake-parse \
            --in-sentences $OUT_FOLDER/${sset}.en \
            --in-actions $OUT_FOLDER/${sset}.actions \
            --entity-rules $OUT_FOLDER/jkaln_2016_scr.entity_rules.json \
            --out-bio-tags $OUT_FOLDER/${sset}_jkaln.tags
    
        python scripts/tasks_from_amr_tags.py \
            $OUT_FOLDER/${sset}_jkaln.tags \
            $OUT_FOLDER/${sset}_jkaln
    
        rm -f $OUT_FOLDER/${sset}.en $OUT_FOLDER/${sset}.actions 
    
    done

done
