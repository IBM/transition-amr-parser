#
# Create Multi-Task Dataset with simulated low ressource from AMR2.0
#
set -o pipefail
set -o errexit
[ -z "$1" ] && echo "$0 out_folder" && exit 1
OUT_FOLDER_ROOT=$1
. set_environment.sh
set -o nounset

[ ! -f DATA/AMR/corpora/amr2.0/train.txt ] && \
    echo "Have you run preprocessing preprocessing/README.md?" && \
    exit 1

# TRAIN DATA
for max_sentences in amr1.0 5000 2500;do

     # Folder for the corpus
     # v3: alignments now computed intra corpus
     OUT_FOLDER=$OUT_FOLDER_ROOT/amr2.0_from_${max_sentences}_v3/
     rm -Rf $OUT_FOLDER
     mkdir -p $OUT_FOLDER 
 
     # Extract entity rules __for the entire set__
     python scripts/extract_rules.py \
         DATA/AMR/corpora/amr2.0/train.txt \
         $OUT_FOLDER/train.entity_rules.json
 
     # The set amr1 is defined, the rest we need to specify sentence number
     if [ "$max_sentences" == "amr1.0" ];then
         extra_flag=""
     else
         extra_flag="--max-sentences $max_sentences"
     fi

    # Split AMR2.0 with o3 alignments into a set with AMR1.0 (or max_sentences)
    # ids and one without. Sampling is made sub-set balanced, based on id names
    python scripts/amr2.0_from_amr1.0.py \
        --in-amr DATA/AMR/corpora/amr2.0/train.txt \
        --in-amr1 DATA/AMR/corpora/amr1.0/train.txt \
        --out-amr-from-amr1 $OUT_FOLDER/amr2.0_from_${max_sentences}_train.txt \
        --out-amr-reminder $OUT_FOLDER/amr2.0_minus_${max_sentences}_train.txt \
        $extra_flag

    # Align the data for each split separately
    # simulated gold
    python preprocess/remove_wiki.py \
        $OUT_FOLDER/amr2.0_from_${max_sentences}_train.txt \
        $OUT_FOLDER/amr2.0_from_${max_sentences}_train.txt.no_wiki
    bash preprocess/align.sh \
        $OUT_FOLDER/amr2.0_from_${max_sentences}_train.txt.no_wiki \
        $OUT_FOLDER/amr2.0_from_${max_sentences}_train_cfill.amr 
    # reminder    
    python preprocess/remove_wiki.py \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train.txt \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train.txt.no_wiki

    bash preprocess/align.sh \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train.txt.no_wiki \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_cfill.amr 

    # Create oracles for the reminder data to extract annotations. For this we
    # can use the global train entity rules
    amr-oracle \
        --in-amr $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_cfill.amr \
        --entity-rules $OUT_FOLDER/train.entity_rules.json \
        --out-sentences $OUT_FOLDER/train.en \
        --out-actions $OUT_FOLDER/train.actions \
        --out-rule-stats $OUT_FOLDER/train.rules.json \
        --copy-lemma-action
    # Create BIO tags from alignments and node actions of that oracle (PRED, ADDNODE)
    amr-fake-parse \
        --in-sentences $OUT_FOLDER/train.en \
        --in-actions $OUT_FOLDER/train.actions \
        --entity-rules $OUT_FOLDER/train.entity_rules.json \
        --out-bio-tags $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_cfill.tags
    
    # Remove oracle
    rm -f $OUT_FOLDER/train.en $OUT_FOLDER/train.actions $OUT_FOLDER/train.rules.json
    
    # Filter PRED, ADDNODE to generate Word Sense Disambiguation WSD data. Rename
    # ADDNODE as Macro (MCR). This includes NER but also othe constructus such as
    # reification
    python scripts/tasks_from_amr_tags.py \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_cfill.tags \
        $OUT_FOLDER/amr2.0_minus_${max_sentences}_train_cfill
    
    # DEV/TEST DATA
    for sset in dev test;do
        
        # Create oracle
        # Note that we can use directly the aligned ones as this is for gold
        # annotations
        amr-oracle \
            --in-amr DATA/AMR/corpora/amr2.0/${sset}.no_wiki.aligned_combo-filled.txt \
            --entity-rules $OUT_FOLDER/train.entity_rules.json \
            --out-sentences $OUT_FOLDER/${sset}.en \
            --out-actions $OUT_FOLDER/${sset}.actions \
            --copy-lemma-action
        # Create BIO tags from alignments and node actions of that oracle (PRED,
        # ADDNODE)
        amr-fake-parse \
            --in-sentences $OUT_FOLDER/${sset}.en \
            --in-actions $OUT_FOLDER/${sset}.actions \
            --entity-rules $OUT_FOLDER/train.entity_rules.json \
            --out-bio-tags $OUT_FOLDER/${sset}_cfill.tags

        # Remove oracle
        rm -f $OUT_FOLDER/${sset}.en $OUT_FOLDER/${sset}.actions 
    
        # Filter PRED, ADDNODE etc
        python scripts/tasks_from_amr_tags.py \
            $OUT_FOLDER/${sset}_cfill.tags \
            $OUT_FOLDER/${sset}_cfill
    
    done

done
