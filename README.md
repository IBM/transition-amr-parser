Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) version `0.5.0`. Current code implements the `Structured-BART` model. The model yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble) on AMR2.0 test. 

Checkout the `action-pointer-transformer` branch (version `0.4.2`) for the `Action Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. The model yields `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test. Due to aligner implementation improvements this code reaches `82.1` on AMR2.0 test.

Checkout the `stack-transformer` branch (version `0.3.4`) for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. This yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

Clone and pip install (see `set_environment.sh` below if you use a virtualenv)

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
. set_environment.sh     # see below
pip install .            # use --editable if to modify code
```

The code needs Pytorch `1.4` and Python `3.6-3.7`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

```bash
# inside set_environment.sh
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```
OR you can leave this empty and handle environment activation yourself i.e.

```bash
touch set_environment.sh
```

train and test scripts always source this script at the beggining i.e.

```bash
. set_environment.sh
```

that will spare you activating the environments or setting up system variables and other each time, which helps when working with computer clusters. 

You will also need Pytorch Scatter version `1.3.2`

```
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
git checkout 1.3.2
pip install .
```

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 1-3 minutes. It wont learn anything but at least will run all stages.
```bash
bash tests/minimal_test.sh
```

If you want to align AMR data, the neural aligner requires a separate allennlp install to extract ELMO

```bash
bash align_cfg/install.sh
```

See [here](scripts/README.md#install-details) for more install details

## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

You will also need to unzip the precomputed BLINK cache

```
unzip /path/to/linkcache.zip
```

To launch train/test use (this will also run the aligner)

```
bash run/run_experiment.sh configs/amr2.0-action-pointer.sh
```

you can check training status with

```
python run/status.py --config configs/amr2.0-action-pointer.sh
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
