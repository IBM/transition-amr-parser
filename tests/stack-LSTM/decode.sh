set -o pipefail 
set -o errexit
# load local variables used below
. set_environment.sh
set -o nounset

# Use the standalone parser
amr-parse \
    --in-sentences ${LDC2016_AMR_CORPUS}/dev.JAMR.tokens \
    --in-model ${LDC2016_AMR_MODELS}/v0.1.0_stack-LSTM/ysook/model.epoch40.params \
    --add-root-token \
    --out-amr dev.amr \
    --batch-size 12 \
    --num-cores 6 \
    --use-gpu

# evaluate model performance
echo "Evaluating Model"
test_result="$(python smatch/smatch.py --significant 3 -f ${LDC2016_AMR_CORPUS}/dev.txt.removedWiki.noempty.JAMRaligned dev.amr -r 10)"
echo $test_result
