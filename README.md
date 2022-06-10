Transition-based AMR Parser
============================

Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR parsing and reliable token to node alignments. See below for the different versions and corresponding papers. Note that, as of now, Structured-BART does not support standalone parsing mode. Use the `action-pointer` branch to get a parser that can work standalone.

### Structured-BART 

Current version (`0.5.2`). Structured-BART [(Zhou et al 2021b)](https://aclanthology.org/2021.emnlp-main.507/) encodes the parser state using specialized cross and self-attention heads and leverages BART's language model to replace the use of subgraph actions and lemmatizer, thus enabling a much simpler oracle with 100% coverage. It yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble). This version introduces the ibm-neural-aligner [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) yielding a base AMR3.0 performance of `82.7` (`83.1` with latent alignment training). Structured-BART is also used for [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) which yields a new single model SoTA of `85.7` for AMR2.0 and `84.1` for AMR3.0 by introducing Smatch-based ensemble distillation.

### Action Pointer

Checkout the `action-pointer` branch (derived from version `0.4.2`) for the `Action Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. As the stack-Transformer, APT encodes the parser state in dedicated attention heads. APT uses however actions creating nodes to represent them. This decouples token and node representations yielding much shorter sequences than previous oracles with higher coverage. APT achieves `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test using RoBERTa embeddings and has an efficient shallow decoder. Due to aligner implementation improvements this code reaches `82.1` on AMR2.0 test, better that what is reported in the paper.

### Stack-Transformer

Checkout the `stack-transformer` branch (derived from version `0.3.4`) for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. The stack-Transformer masks dedicated cross attention heads to encode the parser state represented by stack and buffer. It yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

The code needs Pytorch `1.10` and fairseq `0.10.2`. We tested it with Python `3.6-3.7`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

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

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
git checkout <branch>     # for e.g. action-pointer, ignore for current version
. set_environment.sh      # see above
pip install --editable .   
```

it installs correctly both on OSX and Linux (RHEL). For linux it may be
easier to pre-install with conda targeting your architecture, for example

    conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 10 minutes. It wont learn anything but at least will run all stages.
```bash
bash tests/minimal_test.sh
```

## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

You will also need to unzip the precomputed BLINK cache. See issues in this repository to get the cache file (or the link above for IBM-ers).

```
unzip /path/to/linkcache.zip
```

To launch train/test use (this will also run the aligner)

```
bash run/run_experiment.sh configs/amr2.0-structured-bart-large-sep-voc.sh
```

you can check training status with

```
python run/status.py -c configs/amr2.0-structured-bart-large-sep-voc.sh
```

use `--results` to check for scores once models are finished.

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse -c $in_checkpoint -i $input_file -o file.amr
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
