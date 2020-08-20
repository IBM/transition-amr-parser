Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. The code includes two fundamental components.

1. A State machine and oracle transforming the sequence-to-graph task into a sequence-to-sequence problem. This follows the AMR oracles in [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370).

2. Two structured sequence-to-sequence models able to encode the parse state. This includes stack-LSTM and the stack-Transformer. 

Current version is `0.3.0`. Initial commit developed by Miguel Ballesteros and Austin Blodgett while at IBM. 

## Using the Parser

- The simplest is to use our endpoint (GRPC service) check [services](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Parsing-Services)
- to install through the Watson-NLP artifactory, see the [wiki](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Installing-the-python-package-through-Artifactory)
- to install the parser locally, see below. If you have acess to the CCC there are installers for x86/PPC and pre-trained models available, see the [wiki](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Installing-in-CCC).

Before using the parser, please refer the [Tokenizer](#tokenizer) section on what tokenizer to use.

## Manual Installation

**NOTE:** See if the CCC installers suit your needs first
[wiki](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Installing-in-CCC).
Below are the instructions for a generic architecture.

The code has been tested on Python `3.6`. We use a script to activate
conda/pyenv and virtual environments. If you prefer to handle this yourself
just create an empty file (the scripts will assume it exists in any case)

```bash
touch set_environment.sh
```

then install our modified fairseq

```
. set_environment.sh    # if used
git clone git@github.ibm.com:ramon-astudillo/fairseq.git
cd fairseq
git checkout v0.3.0/decouple-fairseq
pip install .
cd ..
```

the main repo

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
git checkout v0.3.0
pip install .
cd ..
```

and the smatch evaluation tool.

```
git clone https://github.com/snowblink14/smatch.git 
cd smatch
git checkout v1.0.4
cd ..
```

The spacy tools will be updated on first use. To do this manually do

```bash
python -m spacy download en
```

You can check if it worked using

```bash
python tests/correctly_installed.py
```

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse \
  --in-tokenized-sentences $input_file \
  --in-checkpoint $in_checkpoint \
  --out-amr file.amr
```

It will parse each line of `$input_file` separately. `$in_checkpoint` is the
pytorch checkpoint of a trained model. The `file.amr` will contain the PENNMAN
notation AMR with additional alignment information as comments.

To use from other Python code with a trained model do

```python
from transition_amr_parser.stack_transformer_amr_parser import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint) 
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
print(annotations.toJAMRString())
```

## Training your Model

See the CCC training scripts as example `scripts/stack-transformer/jbsub_experiment.sh`
