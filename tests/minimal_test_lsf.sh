set -o errexit 
set -o pipefail
if [ -z $1 ];then 

    # Standard mini-test with wiki25
    config=configs/wiki25-structured-bart-base-neur-al-sampling.sh 

    ELMO_WEIGHTS="DATA/wiki25/aligned/ibm_neural_aligner/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    if [ -f "$ELMO_WEIGHTS" ]; then
        echo "$ELMO_WEIGHTS exists."

        # Backup weights because expensive to download.
        mv $ELMO_WEIGHTS ./tmp.elmo

        # Delete previous runs is exist.
        rm -Rf DATA/wiki25/*

        # Restore elmo weights.
        mkdir -p DATA/wiki25/aligned/ibm_neural_aligner
        mv tmp.elmo $ELMO_WEIGHTS
    else
        echo "$ELMO_WEIGHTS does not exist."

        # Delete previous runs is exist
        rm -Rf DATA/wiki25/*
    fi

    # replace code above with less restrictive deletion
    # rm -R -f DATA/wiki25/embeddings
    # rm -R -f DATA/wiki25/features
    # rm -R -f DATA/wiki25/oracles
    # rm -R -f DATA/wiki25/models

    # simulate completed corpora extraction and alignment
    bash tests/create_wiki25_mockup.sh

else

    # custom config mini-test
    config=$1
fi
set -o nounset 

# Run local test
bash run/lsf/run_experiment.sh $config  

# check if final result is there
. $config

if [ -f "${MODEL_FOLDER}-seed42/beam10/valid_${DECODING_CHECKPOINT}.wiki.smatch" ];then
    printf "\n[\033[92mOK\033[0m] $0\n"
else
    printf "\n[\033[91mFAILED\033[0m] $0\n"
fi
