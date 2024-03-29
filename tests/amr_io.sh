set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# Seems not to be reading 

# passes IO test
python tests/amr_io.py --no-isi \
   --in-amr DATA/AMR2.0/aligned/cofill_isi/dev.txt \
   --ignore-errors 'amr2-dev' \
   # --out-amr tmp.amr

# passes IO test
python tests/amr_io.py --no-isi \
   --in-amr DATA/AMR2.0/aligned/cofill_isi/train.txt \
   --ignore-errors 'amr2-train' \
   # --out-amr tmp.amr

# passes IO test
python tests/amr_io.py --no-isi \
   --in-amr DATA/AMR3.0/aligned/cofill_isi/dev.txt \
   --ignore-errors 'amr3-dev' \
   # --out-amr tmp.amr

# passes IO test
python tests/amr_io.py --no-isi \
   --in-amr DATA/AMR3.0/aligned/cofill_isi/train.txt \
   --ignore-errors 'amr3-train' \
   # --out-amr tmp.amr

printf "[\033[92m OK \033[0m] $0\n"
