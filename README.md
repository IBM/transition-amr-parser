# stack-lstm-amr-parser

Pytorch implementation of a transition-based parser for Abstract Meaning Representation (AMR), based on stack-lstms. 

Current code implements the original stack-lstm parser for AMR from

- Ballesteros, Miguel, and Yaser Al-Onaizan. **AMR Parsing using Stack-LSTMs**. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. (EMNLP 2017)

with some improvements.

Code developed mainly by Miguel Ballesteros and Austin Blodgett while at IBM.

## Install

the code has been tested on Python 3.6

    cd stack-lstm-amr-parser
    pip install -r requirements.txt

you also need to download the spacy English model for lemmatization

    python -m spacy download en

and SMATCH tools to compute scores

    git clone https://github.com/snowblink14/smatch.git

you can use `virtualenv` and `pyenv` or `conda` to virtualize modules and
python versions and work without the need for root

## Data

General training AMR data is available from LDC2017T10. You will need to
reformat the alignments to match the JAMR styles (see sample data file in
`data/wiki25.jkln`). Files in `data/` are provided with CC-SA 4.0 license. We
also provide a sample of the corresponding BERT embeddings in `data/`

### Pre-processing Instructions
After downloading the LDC 2017 data, you can preprocess it as follows. The
scripts will build JAMR and Kevin alignments using the same tokenization and
then merge them together. You must have the following installed: pip, g++, and
ICU (http://site.icu-project.org/home).

```
cd preprocess
bash preprocess.sh path/to/ldc_data
rm train.* dev.* test.*
```
New files will be placed in the `data` folder. The process will take ~1 hour to run.

## Test Run

this will use the sample data (train is same as dev)

```
python learn.py -A data/wiki25.jkaln -a data/wiki25.jkaln -B data/wiki25.bert_max_cased.hdf5 -b data/wiki25.bert_max_cased.hdf5 --name my-model
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
  
## Files

amr.py : contains a basic AMR class and a class JAMR_CorpusReader for reading AMRs from JAMR format.
  
state_machine.py : Implement AMR state machine with a stack and buffer 

data_oracle.py : Implements oracle to assign gold actions.

learn.py : Runs the parser (use `learn.py --help` for options)

stack_lstm.py : Implements Stack-LSTM. 

entity_rules.json : Stores rules applied by the ENTITY action 
