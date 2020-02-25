set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
# Argument handling
config=$1
checkpoint=$2
[ -z "$config" ] && \
    echo -e "\ntest.sh <config> <model_checkpoint>\n" && \
    exit 1
[ -z "$checkpoint" ] && \
    echo -e "\ntest.sh <config> <model_checkpoint>\n" && \
    exit 1
set -o nounset 

# Load config
. "$config"

# fix for ensembles
single_checkpoint=$(echo $checkpoint | sed 's@\.pt:.*@@')
results_folder=$(dirname $single_checkpoint)/$TEST_TAG/
mkdir -p $results_folder

# decode 
echo "fairseq-generate $FAIRSEQ_GENERATE_ARGS --path $checkpoint --results-path $results_folder/valid"
fairseq-generate $FAIRSEQ_GENERATE_ARGS \
    --path $checkpoint \
    --results-path $results_folder/valid
# to profile decoder
# 1. pip install line_profiler
# 2. decorate target function with @profile
# 3. call instead of fairseq-generate
# kernprof -l generate.py $fairseq_generate_args --path $checkpoint
# 4. then you can consult details with 
# python -m line_profiler generate.py.lprof

model_folder=$(dirname $checkpoint)

# Create oracle data
if [ "$TASK_TAG" == "AMR" ];then

    # Create the AMR from the model obtained actions
    amr-fake-parse \
        --in-sentences $ORACLE_FOLDER/dev.en \
        --in-actions $results_folder/valid.actions \
        --out-amr $results_folder/valid.amr \

    # add wiki
    python fairseq/dcc/add_wiki.py \
        $results_folder/valid.amr $WIKI_DEV \
        > $results_folder/valid.wiki.amr

    # Compute score
    smatch.py \
         --significant 4  \
         -f $AMR_DEV_FILE \
         $results_folder/valid.wiki.amr \
         -r 10 \
         > $results_folder/valid.wiki.smatch

    cat $results_folder/valid.wiki.smatch

elif [ "$TASK_TAG" == "NER" ];then

    # play actions to create annotations
    python play.py \
        --in-tokens $ORACLE_FOLDER/dev.en \
        --in-actions $results_folder/valid.actions \
        --machine-type NER \
        --out-annotations-folder $results_folder/ \
        --basename dev
    
    # measure performance
    python bio_tags/metrics.py \
        --in-annotations $results_folder/dev.dat \
        --in-reference-annotations $NER_DEV_FILE \
        --out-score $results_folder/dev.f-measure

elif [ "$TASK_TAG" == "NER+AMR" ];then

    # AMR scores
    python play.py \
        --in-tokens $ORACLE_FOLDER/dev.en \
        --in-actions $results_folder/valid.actions \
        --in-mixing-indices $ORACLE_FOLDER/dev.mixing_indices \
        --out-annotations-folder $results_folder/ \
        --basename dev \
    
    # compute F-measure for NER
    python bio_tags/metrics.py \
        --in-annotations $results_folder/dev.dat \
        --in-reference-annotations $NER_DEV_FILE \
        --out-score $results_folder/dev.f-measure
    cat $results_folder/dev.f-measure

    # compute smatch for AMR
    smatch.py \
     --significant 4  \
         -f $AMR_DEV_FILE \
         $results_folder/dev.amr \
     -r 10 \
         > $results_folder/dev.smatch
    cat $results_folder/dev.smatch

fi
