You will need to install the alignment tools. NOTE: As of this writing,
Pourdamghani's aligner is not more acessible in the usual URL. We suggest
contacting the author for the download (see inside this script).

```bash
bash preprocess/install_alignment_tools.sh
```

You will need first to preprocess the data to obtain the alignments

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR/corpora/amr2.0/
```

NOTE: This are the paths used in experiment configs (see configs/). The same for AMR1.0

```
python preprocess/merge_files.py /path/to/LDC2014T12/data/amrs/split/ DATA/AMR/corpora/amr1.0/
```
