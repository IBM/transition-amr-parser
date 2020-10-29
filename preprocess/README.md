You will need to install the alignment tools. NOTE: As of this writing,
Pourdamghani's aligner is not more acessible in the usual URL. We suggest
contacting the author for the download (see inside this script).

```bash
bash preprocess/install_alignment_tools.sh
```

You will need first to preprocess the data to obtain the alignments. Assuming
your data is located in 

```
LDC_FOLDER=/path/to/abstract_meaning_representation_amr_2.0/data/amrs/split/
```

To compose the different splits into single files for AMR1.0 do

```bash
. set_environment.sh
CORPUS=DATA/AMR/corpora/amr2.0/
mkdir -p $CORPUS
python preprocess/merge_files.py $LDC_FOLDER/training/ $CORPUS/train.txt
python preprocess/merge_files.py $LDC_FOLDER/dev/ $CORPUS/dev.txt 
python preprocess/merge_files.py $LDC_FOLDER/test/ $CORPUS/test.txt
```

NOTE: This are the paths sued in experiment configs (see configs/)

The same for AMR1.0

```
LDC_FOLDER=/path/to/LDC2014T12/data/split/
```

and

```bash
. set_environment.sh
CORPUS=DATA/AMR/corpora/amr1.0/
mkdir -p $CORPUS
python preprocess/merge_files.py $LDC_FOLDER/training/ $CORPUS/train.txt
python preprocess/merge_files.py $LDC_FOLDER/dev/ $CORPUS/dev.txt 
python preprocess/merge_files.py $LDC_FOLDER/test/ $CORPUS/test.txt
```
