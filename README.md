Transition-based AMR Parser
============================

Pytorch implementation of a transition-based parser for Abstract Meaning Representation (AMR). The code includes oracle and state-machine for AMR and an implementation of a stack-LSTM following [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with some improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370). Initial commit developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## Using the Parser

- to install through the Watson-NLP artifactory, see the wiki
- to install the parser manually, see [Manual Install](#manual-install)

Before using the parser, please refer the [Tokenizer](#tokenizer) section on what tokenizer to use.

To use from the command line with a trained model do

```bash
amr-parse \
  --in-sentences /path/to/dev.tokens \
  --in-model /path/to/model.params  \
  --out-amr /path/to/dev.amr \
  --batch-size 12 \
  --parser-chunk-size 128 \
  --num-cores 10 \
  --use-gpu \
  --add-root-token  
```

The argument `--in-sentences` expects whitespace tokenized sentences (one per line). `--batch-size` refers to RoBERTa batch size. `--num-cores` refrees to cpu cores. `--parser-chunk-size` is the batch size for scpu paralelization (RoBERTa not included). The parser expects `<ROOT>` as last token, use `--add-root-token` to do this automatically.

To use from other Python code with a trained model do

```python
from transition_amr_parser import AMRParser

model_path = '/path/to/model.params'
parser = AMRParser(model_path)

tokens = "He wants her to believe in him .".split()
parse = parser.parse_sentence(tokens)
print(parse.toJAMRString())
```

## Manual Install

the code has been tested on Python 3.6. to install

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
# here optionally activate your virtual environment
bash scripts/install.sh
```

This will pip install the repo in `--editable` mode, and download necessary
SpaCy and Smatch tools.

## Training your Model

This assumes that you have acess to the usual AMR training set from LDC
(LDC2017T10). You will need to apply preprocessing to build JAMR and Kevin
Alignments using the same tokenization and then merge them together. You must
have the following installed: pip, g++, and ICU
(http://site.icu-project.org/home).
```bash
cd preprocess
bash preprocess.sh path/to/ldc_data
rm train.* dev.* test.*
```
New files will be placed in the `data` folder. The process will take ~1 hour to run. The call the train script

```
bash scripts/train.sh scripts/local_variables.sh
```

The script `scripts/local_variables.sh` must contain following environment variables

```bash
# amr files
train_file 
dev_file 
# berts in hdf5 (see sample data)
train_bert  
dev_bert 
# experiment data
name 
# hyperparameters
num_cores=10
batch_size=10 
lr=0.005 
```

## Test Run on sample data

We provide annotated examples in `data/` with CC-SA 4.0 license. We also
provide a sample of the corresponding BERT embeddings. This can be used as a
sanity check (but data amount unsufficient for training) . To test training
```
amr-learn -A data/wiki25.jkaln -a data/wiki25.jkaln -B data/wiki25.bert_max_cased.hdf5 -b data/wiki25.bert_max_cased.hdf5 --name toy-model
```

# More information

## Action set

The transition-based parser operates using 10 actions:

  - `SHIFT` : move buffer0 to stack0
  - `REDUCE` : delete token from stack0
  - `CONFIRM` : assign a node concept
  - `SWAP` : move stack1 to buffer
  - `LA(label)` : stack0 parent of stack1
  - `RA(label)` : stack1 parent of stack0
  - `ENTITY(type)` : form a named entity
  - `MERGE` : merge two tokens (for MWEs)
  - `DEPENDENT(edge,node)` : Add a node which is a dependent of stack0
  - `CLOSE` : complete AMR, run post-processing

There are also two optional actions using SpaCy lemmatizer `COPY_LEMMA` and
`COPY_SENSE01`. These actions copy `<lemma>` or `<lemma>-01` to form a node
name.
  
## Files

amr.py : contains a basic AMR class and a class JAMR_CorpusReader for reading AMRs from JAMR format.
  
state_machine.py : Implement AMR state machine with a stack and buffer 

data_oracle.py : Implements oracle to assign gold actions.

learn.py : Runs the parser (use `learn.py --help` for options)

stack_lstm.py : Implements Stack-LSTM. 

entity_rules.json : Stores rules applied by the ENTITY action 

## Tokenizer

For best performance, it is recommended to use the same tokenizer while testing and training. The model works best with the JAMR Tokenizer.

When using the `AMRParser.parse_sentence` method, the parser expects the input to be tokenized words.

When using the parser as a command line interface, the input file must contain 1 sentence per line. Also, generate these sentences by first tokenizing the raw sentences using a tokenizer of your choice and then joining the tokens using whitespace (Since the model just uses whitespace tokenization when called via CLI).


