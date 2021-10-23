Transition-based AMR Parser
============================

Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR alignments and reliable token to node alignments. See below for the different versions and corresponding papers

### Structured-BART 

Current version (`0.5.1`). Structured-BART yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble) on the AMR2.0 test without graph recategorization or external dependencies, excluding wikification. It also produces accurate word to node alignments.See PAPER. As of this writing this is the best AMR parser published as per AMR2.0 test set scores, the standard benchmark.

### Action Pointer

Checkout the `action-pointer` branch (derived from version `0.4.2`) for the `Action Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. APT yields `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test using RoBERTa embeddings and has an efficient shallow decoder. Due to aligner implementation improvements this code reaches `82.1` on AMR2.0 test, better that what is reported in the paper.

### Stack-Transformer

Checkout the `stack-transformer` branch (derived from version `0.3.4`) for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. The stack-Transformer yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

The code needs Pytorch `1.4` and fairseq `0.10.2`. We tested it with Python `3.6-3.7`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

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
git checkout <branch>    # for e.g. action-pointer, ignore for current version
. set_environment.sh     # see above
pip install .            # use --editable if to modify code
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

You will also need to unzip the precomputed BLINK cache. See issues in this repository to get the cache file.

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

As of now `Structured-BART` does not support standalone parsing. Use the
`action-pointer` or `stack-Transformer` branches for this.
