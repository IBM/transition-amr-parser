Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. The code includes two fundamental components.

1. A State machine and oracle transforming the sequence-to-graph task into a sequence-to-sequence problem. This follows the AMR oracles in [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370).

2. Two structured sequence-to-sequence models able to encode the parse state. This includes stack-LSTM and the stack-Transformer. 

Current version is `0.3.2`. Initial commit developed by Miguel Ballesteros and Austin Blodgett while at IBM. Current [contributors](https://github.ibm.com/mnlp/transition-amr-parser/graphs/contributors).

## Using the Parser

- To test an idea for few sentences, use our web interfaces, see [services](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Parsing-Services)
- In the same page you have GRPC service URLs that require only a minor client side install 
- to install the parser locally, see below. If you have acess to the CCC there are installers for x86/PPC and pre-trained models available, see the [wiki](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Installing-in-CCC).

Note the the parser consumes word-tokenized text. It is not greatly affected by
different tokenizers. We reccomend to use the 1NLP tokenizer.

## Manual Installation

**NOTE:** See if the CCC installers suit your needs first
[wiki](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Installing-in-CCC).
Below are the instructions for `v0.3.2`

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
git checkout v0.3.2
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

Then use either `conda` or `pip` to install. For `conda` do

```
conda env update -f scripts/stack-transformer/environment.yml
pip install spacy==2.2.3 smatch==1.0.4 ipdb
```

For `pip` only do (ignore if you use the ones above)

```
pip install -r scripts/stack-transformer/requirements.txt
```

Then download and patch fairseq using

```
bash scripts/download_and_patch_fairseq.sh
```

Finally install fairseq without dependencies (installed above) and this repo.
The `--editable` flag allows to modify the code without the need to reinstall.

```
pip install --no-deps --editable fairseq-stack-transformer-v0.3.2
pip install --editable .
```

The spacy tools will be updated on first use. You can force this manually with 

```bash
python -m spacy download en
```

To check if install worked do

```bash
python tests/correctly_installed.py
```

If you are installing in PowerPCs, you will have to use the conda option. Also
spacy has to be installed with conda instead of pip (2.2.3 version will not be
available, which affects the lematizer behaviour)

## Training a model

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
will contain the PENNMAN notation AMR with additional alignment information as
comments.

To use from other Python code with a trained model do

```python
from transition_amr_parser.stack_transformer_amr_parser import AMRParser
parser = AMRParser.from_checkpoint(in_checkpoint) 
annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
print(annotations.toJAMRString())
```

## Training your Model

See the CCC training scripts as example `scripts/stack-transformer/jbsub_experiment.sh`
