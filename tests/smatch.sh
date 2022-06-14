set -o errexit
set -o pipefail
. set_environment.sh
set -o nounset

# DOES NOT PASS Smatch test (due to Smatch read BUGs)
# :mod 277703234 in dev[97]
python scripts/smatch_aligner.py \
    --in-amr DATA/AMR2.0/corpora/dev.txt \
    --in-reference-amr DATA/AMR2.0/corpora/dev.txt \
    # --stop-if-different

exit

# DOES NOT PASS Smatch test (due to Smatch read BUGs)
# bolt12_10511_2844.2 ignores :mod "A"
# bolt12_12120_6501.3 ignores b2 :mod 106
# bolt12_12120_6501.4 ignores b :mod 920
# bolt12_12120_6501.5 ignores b :mod 17, b :mod 14
# ... (stopped counting)
python scripts/smatch_aligner.py \
    --in-amr DATA/AMR2.0/corpora/train.txt \
    --in-reference-amr DATA/AMR2.0/corpora/train.txt \
    --stop-if-different

# DOES NOT PASS Smatch test (due to Smatch read BUGs)
# :mod 277703234 in dev[97]
python scripts/smatch_aligner.py \
    --in-amr DATA/AMR2.0/corpora/dev.txt \
    --in-reference-amr DATA/AMR2.0/corpora/dev.txt \
    --stop-if-different

# DOES NOT PASS Smatch test (due to Smatch read BUGs)
# bolt12_10511_2844.2 ignores :mod "A"
# bolt12_12120_6501.3 ignores b2 :mod 106
# bolt12_12120_6501.4 ignores b :mod 920
# bolt12_12120_6501.5 ignores b :mod 17, b :mod 14
# ... (stopped counting)
python scripts/smatch_aligner.py \
    --in-amr DATA/AMR2.0/corpora/train.txt \
    --in-reference-amr DATA/AMR2.0/corpora/train.txt \
    --stop-if-different
