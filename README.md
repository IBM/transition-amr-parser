Transition-based AMR Parser
============================

Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR parsing and reliable token to node alignments. See below for the different versions and corresponding papers. Note that, as of now, Structured-BART does not support standalone parsing mode. Use the `action-pointer` branch to get a parser that can work standalone.

### Structured-mBART

Current version. Cross-lingual version of Structured-BART

### Structured-BART 

Structured-BART [(Zhou et al 2020b)](https://openreview.net/forum?id=qjDQCHLXCNj) encodes the parser state using specialized cross and self-attention heads and leverages BART's language model to replace the use of subgraph actions and lemmatizer, thus enabling a much simpler oracle with 100% coverage. Its yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble) on the AMR2.0 test-set without graph recategorization or external dependencies, excluding wikification. It also produces accurate word to node alignments. As of this writing, this is the best AMR parser published as per AMR2.0 test set scores, the standard benchmark.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

The code needs Pytorch `1.6` and fairseq `0.10.2`. We tested it with Python `3.8`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

```bash
# inside set_environment.sh
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```
OR you can leave this empty and handle environment activation yourself i.e.

```bash
touch set_environment.sh
```

Note that all bash scripts always source `set_environment.sh`, so you do not need to source it yourself.

To install clone and pip install
torch-scatter should be installed on a GPU machine unless you know how to do it without GPU:)

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git --branch Structured-mBART -s Structured-mBART
cd transition-amr-parser
git checkout <branch>    # for e.g. action-pointer, ignore for current version
. set_environment.sh     # see above
pip install -e .            # remove -e(ditable) option if you do not need to modifiy the source code
pip install torch-scatter==1.3.2
```

Download mbart and place it in DATA/ directory
```bash
cd DATA
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz;
tar -xzvf mbart.cc25.v2.tar.gz
cd mbart.cc25.v2; ln -s sentence.bpe.model sentencepiece.bpe.model
```

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 10 minutes. It wont learn anything but at least will run all stages.
```bash
bash tests/minimal_test.sh
```


If you want to align AMR data, the aligner uses additional tools that can be donwloaded and installed with

```bash
bash preprocess/install_alignment_tools.sh
```

See [here](scripts/README.md#install-details) for more install details

## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

To launch train/test use (this will also run the aligner) for English AMR2.0

```
bash run/run_experiment.sh configs/amr2en.sh
```

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse -c $in_checkpoint -i $input_file -o file.amr --roberta-cache-path DATA/mbart.cc25.v2
```

It will parse each line of `$input_file` separately (assumed tokenized).
`$in_checkpoint` is the Pytorch checkpoint of a trained model. The `file.amr`
will contain the PENMAN notation AMR with additional alignment information as
comments. Use the flag `--service` together with `-c` for an iterative parsing
mode.

To use from other Python code with a trained model do

```python
from transition_amr_parser.parse import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint)
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
# Penman notation
print(''.join(annotations[0][0]))
```
