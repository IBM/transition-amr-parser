set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# # AMR2.0
# python tests/amr_io.py \
#    --in-amr DATA/AMR2.0/corpora/train.txt \
#    --ignore-errors 'amr2-train'
#    --out-amr tmp.txt
# 
# smatch_score=$(smatch.py -r 10 --significant 4  \
# -f DATA/AMR2.0/corpora/train.txt \
# tmp.amr \
# )
# echo "$smatch_score"
# rm tmp.txt

python tests/amr_io.py \
   --in-amr DATA/AMR2.0/corpora/dev.txt \
   --ignore-errors 'amr2-dev' \
   --out-amr tmp.amr

python scripts/smatch_aligner.py --in-amr tmp.amr --in-reference-amr DATA/AMR2.0/corpora/dev.txt --stop-if-different
exit

smatch_score=$(python smatch/smatch.py -r 10 --significant 4  \
-f DATA/AMR2.0/corpora/dev.txt \
tmp.amr \
)
echo "$smatch_score"
exit
rm tmp.txt


# AMR3.0
python tests/amr_io.py \
   --in-amr DATA/AMR3.0/corpora/train.txt \
   --ignore-errors 'amr3-train'
python tests/amr_io.py \
   --in-amr DATA/AMR3.0/corpora/dev.txt \
   --ignore-errors 'amr3-dev'

printf "[\033[92mOK\033[0m] AMR class passes\n"
