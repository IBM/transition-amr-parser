set -o errexit
set -o pipefail
set -o nounset
amr-edit --in-amr /dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned --out-amr dev.txt.removedWiki.noempty.JAMRaligned
vimdiff /dccstor/ykt-parse/SHARED/CORPORA/AMR/LDC2016T10_preprocessed_tahira/dev.txt.removedWiki.noempty.JAMRaligned dev.txt.removedWiki.noempty.JAMRaligned
