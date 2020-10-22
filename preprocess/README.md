You will need to install the alignment tools. NOTE: As of this writing,
Pourdamghani's aligner is not more acessible in the usual URL. We suggest
contacting the author for the download (see inside this script).

```bash
bash preprocess/install_alignment_tools.sh
```

You will need first to preprocess the data to obtain the alignments. Assuming
your data is located in 

```
LDC_FOLDER=/path/to/amr2.0/data/amrs/split/
```

To compose the different splits into single files do

```bash
. set_environment.sh
# NOTE: This is the path set in experiment configs
CORPUS=DATA/AMR/corpora/amr2.0/
mkdir -p $CORPUS
python preprocess/merge_files.py $LDC_FOLDER/training/ $CORPUS/train.txt
python preprocess/merge_files.py $LDC_FOLDER/dev/ $CORPUS/dev.txt 
python preprocess/merge_files.py $LDC_FOLDER/test/ $CORPUS/test.txt
```

then remove wiki

```bash
python preprocess/remove_wiki.py $CORPUS/train.txt $CORPUS/train.no_wiki.txt
python preprocess/remove_wiki.py $CORPUS/dev.txt $CORPUS/dev.no_wiki.txt
python preprocess/remove_wiki.py $CORPUS/test.txt $CORPUS/test.no_wiki.txt
```

and finally align the files (this can take around 1h)

```bash
bash preprocess/align.sh $CORPUS/train.no_wiki.txt $CORPUS/train.no_wiki.aligned.txt
bash preprocess/align.sh $CORPUS/dev.no_wiki.txt $CORPUS/dev.no_wiki.aligned.txt
bash preprocess/align.sh $CORPUS/test.no_wiki.txt $CORPUS/test.no_wiki.aligned.txt
```
