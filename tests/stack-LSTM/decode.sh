set -o pipefail 
set -o errexit
# load local variables used below
. set_environment.sh
set -o nounset

mkdir -p DATA.test

# Use the standalone parser
amr-parse \
    --in-sentences ${LDC2016_AMR_CORPUS}/dev.JAMR.tokens \
    --in-model ${LDC2016_AMR_MODELS}/v0.1.0_stack-LSTM/ysook/model.epoch40.params \
    --add-root-token \
    --out-amr DATA.test/dev_v0.1.0_stack-LSTM.amr \
    --batch-size 12 \
    --num-cores 6 \
    --use-gpu

# evaluate model performance
echo "Evaluating Model"
ref_smatch=0.733
smatch_args="
    --significant 3 
    -r 10 
    -f 
        ${LDC2016_AMR_CORPUS}/dev.txt.removedWiki.noempty.JAMRaligned 
        DATA.test/dev_v0.1.0_stack-LSTM.amr"
smatch="$(smatch.py $smatch_args)"

echo "$smatch"
echo "$ref_smatch"

if [ "$smatch" != "F-score: $ref_smatch" ];then
    echo -e "[\033[91mFAILED\033[0m] stack-LSTM decode F-score not $ref_smatch"
    exit 1
else
    echo -e "[\033[92mOK\033[0m] stack-LSTM decode test passed!"
fi
