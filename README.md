Transition-based Neural Parser
============================

State-of-the-Art Abstract Meaning Representation (AMR) parsing, see [papers
with code](https://paperswithcode.com/task/amr-parsing). Models both
distribution over graphs and aligments with a transition-based approach. Parser
supports any other graph formalism as long as it is expressed in [Penman
notation](https://penman.readthedocs.io/en/latest/notation.html).

Some of the main features

- [Smatch](https://github.com/snowblink14/smatch) wrapper providing [significance testing](scripts/README.md#paired-boostrap-significance-test-for-Smatch) for Smatch and [MBSE](scripts/README.md#maximum-bayesian-smatch-ensemble-mbse) ensembling.
- `Structured-BART` [(Zhou et al 2021b)](https://aclanthology.org/2021.emnlp-main.507/) with [trained checkpoints](#available-pretrained-model-checkpoints) for document-level AMR [(Naseem et al 2022)](https://aclanthology.org/2022.naacl-main.256), MBSE [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) and latent alignments training [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464)
- `Structured-mBART` for multi-lingual support (EN, DE, Zh, IT) [(Lee et al 2022)](https://arxiv.org/abs/2112.07790)
- Action-Pointer Transformer (`APT`) [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443), checkout `action-pointer` branch 
- `Stack-Transformer` [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89), checkout `stack-Transformer` branch

## Install Instructions

create and activate a virtual environment, for example conda with python 3.8 (we also tested 3.7, 3.9)

```
conda create -y -p ./cenv_x86 python=3.8
conda activate ./cenv_x86
```

or alternatively use `virtualenv`. Note that all scripts source a
`set_environment.sh` script that you can use to activate your virtual
environment as above and set environment variables. If not used, just create 
an empty version

```
# or e.g. put inside conda activate ./cenv_x86
touch set_environment.sh
```

Then install the parser package using pip. You will need to install
`torch-scatter` by separate since it is custom built for CUDA. Here we specify the
call for `torch 1.13.1` and `cuda 11.7`. See [torch-scatter
repository](https://pypi.org/project/torch-scatter/) to find the appropriate
installation instructions.

```
pip install transition-neural-parser
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

If you plan to edit the code, clone and install instead

```
# clone this repo (see link above), then
cd transition-neural-parser
pip install --editable .
```

If you want to train a document-level AMR parser you will also need 

```
git clone https://github.com/IBM/docAMR.git
cd docAMR
pip install .
cd ..
```

## Parse with a pretrained model

Here is an example of how to download and use a pretrained AMR parser in Python

```python
from transition_amr_parser.parse import AMRParser

# Download and save a model named AMR3.0 to cache
parser = AMRParser.from_pretrained('AMR3-structbart-L')
tokens, positions = parser.tokenize('The girl travels and visits places')

# Use parse_sentence() for single sentences or parse_sentences() for a batch
annotations, machines = parser.parse_sentence(tokens)

# Print Penman notation
print(annotations)

# Print Penman notation without JAMR, with ISI
amr = machines.get_amr()
print(amr.to_penman(jamr=False, isi=True))

# Plot the graph (requires matplotlib)
amr.plot()

```

Note that Smatch does not support ISI-type alignments and gives worse results.
Set `isi=False` to remove them. 

You can also use the command line to run a pretrained model to parse a file:

```bash
amr-parse -c $in_checkpoint -i $input_file -o file.amr
```

Download models can invoked with `-m <config>` can be used as well.

Note that Smatch does not support ISI and gives worse results. Use `--no-isi`
to store alignments in `::alignments` meta data. Also use `--jamr` to add JAMR
annotations in meta-data. Use `--no-isi` to store alignments in `::alignments`
meta data. Also use `--jamr` to add JAMR annotations in meta-data.

## Document-level Parsing

This represents co-reference using `:same-as` edges. To change
the representation and merge the co-referent nodes as in the paper, please refer
to [the DocAMR repo](https://github.com/IBM/docAMR.git)

```python
from transition_amr_parser.parse import AMRParser

# Download and save the docamr model to cache
parser = AMRParser.from_pretrained('doc-sen-conll-amr-seed42')

# Sentences in the doc
doc = ["Hailey likes to travel." ,"She is going to London tomorrow.", "She will walk to Big Ben when she goes to London."]

# tokenize sentences if not already tokenized
tok_sentences = []
for sen in doc:
    tokens, positions = parser.tokenize(sen)
    tok_sentences.append(tokens)

# parse docs takes a list of docs as input
annotations, machines = parser.parse_docs([tok_sentences])

# Print Penman notation
print(annotations[0])

# Print Penman notation without JAMR, with ISI
amr = machines[0].get_amr()
print(amr.to_penman(jamr=False, isi=True))

# Plot the graph (requires matplotlib)
amr.plot()

```

To parse a document from the command line the input file `$doc_input_file` is a
text file where each line is a sentence in the document and there is a newline
('\n') separating every doc (even at the end) 


```bash
amr-parse -c $in_checkpoint --in-doc $doc_input_file -o file.docamr --sliding
```

This will output a `.force_actions` file to the same directory as input ,
containing the actions needed to force sentence ends in the document as well as
the output docamr `file.docamr`

## Available Pretrained Model Checkpoints

The models downloaded using `from_pretrained()` will be stored to the pytorch
cache folder under:
```python
cache_dir = torch.hub._get_torch_home()
```

This table shows you available pretrained model names to download;

| pretrained model name      | corresponding file name                               | paper                                                           | beam10-Smatch |
|:--------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------:|:-------------:|
| AMR3-structbart-L-smpl     | amr3.0-structured-bart-large-neur-al-sampling5-seed42 | [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) PR     | 82.9 (beam1)  |
| AMR3-structbart-L          | amr3.0-structured-bart-large-neur-al-seed42           | [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) MAP    | 82.6          |
| AMR2-structbart-L          | amr2.0-structured-bart-large-neur-al-seed42           | [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) MAP    | 84.0          |
| AMR2-joint-ontowiki-seed42 | amr2joint_ontowiki2_g2g-structured-bart-large-seed42  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 85.9          |
| AMR2-joint-ontowiki-seed43 | amr2joint_ontowiki2_g2g-structured-bart-large-seed43  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 85.9          |
| AMR2-joint-ontowiki-seed44 | amr2joint_ontowiki2_g2g-structured-bart-large-seed44  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 85.9          |
| AMR3-joint-ontowiki-seed42 | amr3joint_ontowiki2_g2g-structured-bart-large-seed42  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 84.4          |
| AMR3-joint-ontowiki-seed43 | amr3joint_ontowiki2_g2g-structured-bart-large-seed43  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 84.4          |
| AMR3-joint-ontowiki-seed44 | amr3joint_ontowiki2_g2g-structured-bart-large-seed44  | [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | 84.4          |
| doc-sen-conll-amr-seed42   | both_doc+sen_trainsliding_ws400x100-seed42            |                                                                 |               |

we also provide the trained `ibm-neural-aligner` under names
`AMR2.0_ibm_neural_aligner.zip` and `AMR3.0_ibm_neural_aligner.zip`. For the
ensemble we provide the three seeds. Following fairseq conventions, to run the
ensemble just give the three checkpoint paths joined by `:` to the normal
checkpoint argument `-c`. Note that the checkpoints were trained with the
`v0.5.1` tokenizer, this reduces performance by `0.1` on `v0.5.2` tokenized
data.

Note that we allways report average of three seeds in papers while these are
individual models. A fast way to test models standalone is

    bash tests/standalone.sh configs/<config>.sh

## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
conda activate ./cenv_x86 # activate parser environment
python scripts/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

You will also need to unzip the precomputed BLINK cache. See issues in this repository to get the cache file (or the link above for IBM-ers).

```
unzip /path/to/linkcache.zip
```

To launch train/test use (this will also run the aligner)

```
bash run/run_experiment.sh configs/amr2.0-structured-bart-large.sh
```

Training will store and evaluate all checkpoints by default (see config's
`EVAL_INIT_EPOCH`) and select the one with best dev Smatch. This needs a lot of
space but you can launch a parallel job that will perform evaluation and delete
Checkpoints not in the top `5` 

```
bash run/run_model_eval.sh configs/amr2.0-structured-bart-large.sh
```

you can check training status with

```
python run/status.py -c configs/amr2.0-structured-bart-large.sh
```

use `--results` to check for scores once models are finished.

We include code to launch parallel jobs in the LSF job schedules. This can be
adapted for other schedulers e.g. Slurm, see [here](run/lsf/README.md)

## Initialize with WatBART

To load WatBART instead of BART just uncomment and provide the path on

```
initialize_with_watbart=/path/to/checkpoint_best.pt
```
