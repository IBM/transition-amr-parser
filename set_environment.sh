# This file sets up the basic python installation and environment variables.
# Its a copy of
#
# /dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/set_environment.sh
#
# Do not commit this file! it contains system/local specific settings and paths

# Architecture dependent config
# set conda path for each architecture if needed

export CONDA_DIR=/dccstor/multi-parse/anaconda3
export CONDA_DIR_PPC=/dccstor/multi-parse/miniconda3
export IBM_POWERAI_LICENSE_ACCEPT=yes

if [[ "$HOSTNAME" =~ dccpc.* ]];then
    echo "PowerPC Machine"
    eval "$(${CONDA_DIR_PPC}/bin/conda shell.bash hook)"

    # Create env if missing
    [ ! -d cenv_ppc ] && conda create -y -p ./cenv_ppc 
    echo "conda activate ./cenv_ppc"
    conda activate ./cenv_ppc 

elif [[ "$HOSTNAME" =~ dccx[cn].* ]];then
    echo "x86 Machine"
    eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
    # In CCC this is needed to use old CUDA versions
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cuda-9.1/x86_64/lib64

    # Create env if missing
    #[ ! -d cenv_x86 ] && conda create -y -p ./cenv_x86 
    #echo "conda activate ./cenv_x86"
    #conda activate ./cenv_x86 

    echo ". venv/bin/activate"
    . venv/bin/activate

else
    echo -e "\nhost $HOSTNAME does not match CCC computing nodes (dccxc.*|dccpc.*)\n"
    #exit
fi

# exit if conda not found
#type conda >/dev/null 2>&1 || { echo >&2 "conda not found on PATH"; exit 1; }

# this is neded unless we install
export PYTHONPATH=.

# Paths to used data
# note that training data is a LDC2016 preprocessed using Tahira's scripts
LDC2016_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/
# This contains 2017 data preprocessed follwoing transition_amr_parser README
# it is needed to pass the dev oracle tests
LDC2017_AMR_CORPUS=/dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2017T10_preprocessed_TAP_v0.0.1/

# Path to trained models
LDC2016_AMR_MODELS=/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/
