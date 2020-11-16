set -o errexit 
set -o pipefail
# setup environment
. set_environment.sh
# Argument handling
config=$1
checkpoint=$2
results_path=$3
HELP="\ntest.sh <config> <model_checkpoint> [<results_path>]\n"
[ -z "$config" ] && echo -e $HELP && exit 1
[ -z "$checkpoint" ] && echo -e $HELP && exit 1
[ -z "$results_path" ] && results_path=""
set -o nounset 

# Load config
. "$config"

# Default path
if [ "$results_path" == "" ];then
    # fix for ensembles
    single_checkpoint=$(echo $checkpoint | sed 's@\.pt:.*@@')
    results_path=$(dirname $single_checkpoint)/$TEST_TAG/valid
fi
mkdir -p $(dirname $results_path)

# to profile decoder
# decorate target function with @profile
# test_command="kernprof -o generate.lprof -l fairseq/generate.py"
# python -m line_profiler generate.py.lprof
test_command=fairseq-generate

# Decode to get predicted action sequence
if [ ! -f "${results_path}.actions" ];then
    echo "$test_command $FAIRSEQ_GENERATE_ARGS --path $checkpoint --results-path ${results_path}"
    $test_command $FAIRSEQ_GENERATE_ARGS \
        --path $checkpoint \
        --results-path ${results_path} 
fi

# Call state machine(s) with action sequence. For multi-task separate tasks
# into different files
if [ "$TASK_TAG" == "AMR" ];then

    # AMR
    amr-fake-parse \
        --entity-rules $ENTITY_RULES \
        --in-sentences $ORACLE_FOLDER/dev.en \
        --in-actions ${results_path}.actions \
        --out-amr ${results_path}.amr 

elif [ "$TASK_TAG" == "dep-parsing" ];then

    # dependency parsing
    # No need for playing state-machine
    echo ""

fi

# FOR EACH TASK EVALUATE FOR EACH OF THE SUB TASKS INVOLVED e.g. AMR+NER
for single_task  in $(python -c "print(' '.join('$TASK_TAG'.split('+')))");do

    if [ "$single_task" == "AMR" ];then
    
        # AMR (Smatch)
        # Create the AMR from the model obtained actions
        if [ "$WIKI_DEV" == "" ];then
            echo "$WIKI_DEV"
    
            # Smatch evaluation without wiki
            # Compute score in the background
            python smatch/smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE \
                 ${results_path}.amr \
                 -r 10 \
                 > ${results_path}.smatch
            # plot score
            cat ${results_path}.smatch
            
        else
    
            # Smatch evaluation with wiki
            # add wiki
#            python scripts/add_wiki.py \
#                ${results_path}.amr $WIKI_DEV \
#                > ${results_path}.wiki.amr
            python scripts/retyper.py \
                --inputfile ${results_path}.amr \
                --outputfile ${results_path}.wiki.amr \
                --skipretyper \
                --wikify \
                --blinkcachepath $BLINK_CACHE_PATH \
                --blinkthreshold 0.0

            # Compute score in the background
            smatch.py \
                 --significant 4  \
                 -f $AMR_DEV_FILE_WIKI \
                 ${results_path}.wiki.amr \
                 -r 10 \
                 > ${results_path}.wiki.smatch
            # plot score
            cat ${results_path}.wiki.smatch
        
        fi

    elif [ "$single_task" == "dep-parsing" ];then
    
        # dep-parsing (UAS/LAS)
        python scripts/dep_parsing_score.py \
            --in-tokens $ORACLE_FOLDER/dev.en \
            --in-actions ${results_path}.actions \
            --in-gold-actions $ORACLE_FOLDER/dev.actions \
            > ${results_path}.las
        cat ${results_path}.las

    else
 
        echo "Unsupported task $single_task"
        exit 1

    fi
 
done
