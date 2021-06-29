Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch version `0.4.2`. Current code implements the `Action-Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. This model yields `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test.

Checkout the stack-transformer branch for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. This yields `80.5` Smatch (`81.3` with self-learning) on AMR2.0 test. Used in our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

Just clone and pip install (see `set_environment.sh` below if you use a virtualenv)

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
pip install .  # use --editable if you plan to modify code
```

We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

```bash
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```
You can leave this empty if you don't want to use it

```bash
touch set_environment.sh
```

train and test scripts always source this script i.e.

```bash
. set_environment.sh
```

that will spare you activating the environments or setting up system variables and other each time, which helps when working with computer clusters. 

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 1-3 minutes. It wont learn anything but at least will run all stages.
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

You will also need to unzip the precomputed BLINK cache

```
unzip /path/to/linkcache.zip
```

To launch train/test use

```
bash run/run_experiment.sh configs/amr2.0-action-pointer.sh
```

you can check training status with

```
python run/status.py --config configs/amr2.0-action-pointer.sh
```

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse -c $in_checkpoint -i $input_file -o file.amr
```

It will parse each line of `$input_file` separately (assumed tokenized).
`$in_checkpoint` is the Pytorch checkpoint of a trained model. The `file.amr`
will contain the PENMAN notation AMR with additional alignment information as
comments.

To use from other Python code with a trained model do

```python
from transition_amr_parser.parse import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint)
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
print(annotations.toJAMRString())
```
