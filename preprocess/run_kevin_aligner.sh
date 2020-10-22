set -o errexit 
set -o pipefail
[ "$#" -lt 3 ] && echo "e.g. $0 amrs sents out" && exit 1
amrs=$1
sents=$2
outfile=$3
set -o nounset

cd preprocess/kevin/

# virtualenv
if [ ! -d python2-env ];then 
    pip install virtualenv
    virtualenv -p /usr/bin/python2.7 python2-env
fi
. python2-env/bin/activate

echo -e "AMR=$amrs\nENG=$sents\nMGIZA_SCRIPT=mgiza/mgizapp/scripts\nMGIZA_BIN=mgiza/mgizapp/bin\n" > addresses.keep
bash run.sh 
cp AMR_Aligned.keep $outfile
deactivate
cd ../..
