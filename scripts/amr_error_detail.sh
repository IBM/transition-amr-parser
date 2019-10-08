#
# Computes detailed error analysis given various parser results stored in data/system_outputs/
#

set -o nounset
set -o pipefail 
set -o errexit 

[ ! -d scripts/ ] && echo "Call as scripts/$(basename $0)" && exit 1
[ ! -d data/system_outputs ] && mkdir -p data/system_outputs
[ ! -d data/system_reports ] && mkdir -p data/system_reports

# Python3 (watch out Python2 is used at the end of the script)
. venv/bin/activate
. scripts/local_variables.sh

# if not available, compute AMR from current oracle
if [ ! -f data/system_outputs/dev.txt.oracle.amr ];then
    echo "Writing oracle AMR to data/system_outputs/dev.txt.oracle.amr"
    python data_oracle.py $dev_file \
        data/system_outputs/dev.txt.oracle.amr \
        data/system_reports/dev.txt.oracle.actions \
        > /dev/null 2>&1
fi

out_files=$(ls data/system_outputs/)

# For each systems output, compute stats
for amr_output in $out_files;do

     echo $amr_output
     echo "Smatch: data/system_reports/${amr_output}.smatch"
     python smatch/smatch.py \
          --significant 4  \
          -f $dev_file \
          data/system_outputs/$amr_output \
          -r 10 \
          > data/system_reports/${amr_output}.smatch
  
     echo "Smatch per file: data/system_reports/${amr_output}.smatch_sentences"
     python smatch/smatch.py \
          --significant 4  \
          -f $dev_file \
          data/system_outputs/$amr_output \
          -r 10 \
          --ms \
          > data/system_reports/${amr_output}.smatch_sentences

    echo ""
  
done

# For each systems output, compute detailed stats
# NOTE: needs python2 (I use pyenv to have folder-dependent python version,
# see dcc/install.sh)
cd amr-evaluation
. venv_py2/bin/activate
for amr_output in $out_files;do

     echo $amr_output
     echo "Smatch detail: data/system_reports/$(basename ${amr_output}).amr_detail"
     bash ./evaluation.sh \
         ../data/system_outputs/$amr_output $dev_file \
         > ../data/system_reports/$(basename ${amr_output}).amr_detail
     echo ""
            
done
