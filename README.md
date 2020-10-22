Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. The code includes two fundamental components.

1. A State machine and oracle transforming the sequence-to-graph task into a sequence-to-sequence problem. This follows the AMR oracles in [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370) and [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde)

2. Two structured sequence-to-sequence models able to encode the parser state. This includes stack-LSTM [(Dyer et al)](https://arxiv.org/pdf/1505.08075.pdf) and the stack-Transformer [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde). 

Current version is `0.3.3rc`. Initial commit developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## Manual Installation

Clone the repository

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
```

The code has been tested on Python `3.6.9`. We use a script to activate
conda/pyenv and virtual environments. If you prefer to handle this yourself
just create an empty file (the training scripts will assume it exists in any
case).

```bash
touch set_environment.sh
# inside: your source venv/bin/activate or conda activate ./cenv
. set_environment.sh
```

Then for `pip` only install do

```
pip install -r scripts/stack-transformer/requirements.txt
bash scripts/download_and_patch_fairseq.sh
pip install --no-deps --editable fairseq-stack-transformer-v0.3.3
pip install --editable .
```

Alternatively for a `conda` install do

```
conda env update -f scripts/stack-transformer/environment.yml
pip install spacy==2.2.3 smatch==1.0.4 ipdb
bash scripts/download_and_patch_fairseq.sh
pip install --no-deps --editable fairseq-stack-transformer-v0.3.3
pip install --editable .
```

This code will download and patch fairseq before installing. The `--editable`
flag allows to modify the code without the need to reinstall. The spacy tools
will be updated on first use. You can force this manually with 

```bash
python -m spacy download en
```

To check if install worked do

```bash
. set_environment.sh
python tests/correctly_installed.py
```

If you are installing in PowerPCs, you will have to use the conda option. Also
spacy has to be installed with conda instead of pip (2.2.3 version will not be
available, which affects the lematizer behaviour)

## Training a model

You first need to preprocess and align the data. See `preprocess/README.md`. 

Then just call a config to carry a desired experiment

```bash
bash scripts/stack-transformer/experiment.sh scripts/stack-transformer/configs/amr2_o5+Word100_roberta.large.top24_stnp6x6.sh
```

Note that there is cluster version of this script, currently only supporting
LSF but easily adaptable.

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse \
  --in-tokenized-sentences $input_file \
  --in-checkpoint $in_checkpoint \
  --out-amr file.amr
```

It will parse each line of `$input_file` separately (assumed tokenized).
`$in_checkpoint` is the pytorch checkpoint of a trained model. The `file.amr`
will contain the PENMAN notation AMR with additional alignment information as
comments.

To use from other Python code with a trained model do

```python
from transition_amr_parser.stack_transformer_amr_parser import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint) 
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
print(annotations.toJAMRString())
```
