Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. The code includes two fundamental components.

1. A State machine and oracle transforming the sequence-to-graph task into a sequence-to-sequence problem. This follows the AMR oracles in [(Ballesteros and Al-Onaizan 2017)](https://arxiv.org/abs/1707.07755v1) with improvements from [(Naseem et al 2019)](https://arxiv.org/abs/1905.13370) and [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde)

2. The stack-Transformer [(Fernandez Astudillo et al 2020)](https://openreview.net/pdf?id=b36spsuUAde). A sequence to sequence model that also encodes stack and buffer state of the parser into its attention heads.

Current version is `0.3.3` and yields `80.5` Smatch on the AMR2.0 test-set using the default stack-Transformer configuration. Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

Check [Parsing Services](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Parsing-Services) for the endpoint URLs and Docker instructions. If you have acess to CCC and LDC data, we have available both the train data and trained models.

## Manual Installation

Clone the repository

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
```

The code has been tested on Python `3.6` and `3.7` (x86 only). We use a script
to activate conda/pyenv and virtual environments. If you prefer to handle this
yourself just create an empty file (the training scripts will assume it exists
in any case).

```bash
touch set_environment.sh
```

Then for `pip` only install do

```
. set_environment.sh
pip install -r scripts/stack-transformer/requirements.txt
bash scripts/download_and_patch_fairseq.sh
pip install --no-deps --editable fairseq-stack-transformer
pip install --editable .
```

Alternatively for a `conda` install do

```
. set_environment.sh
conda env update -f scripts/stack-transformer/environment.yml
pip install spacy==2.2.3 smatch==1.0.4 ipdb
bash scripts/download_and_patch_fairseq.sh
pip install --no-deps --editable fairseq-stack-transformer
pip install --editable .
```

If you are installing in PowerPCs, you will have to use the conda option. Also
spacy has to be installed with conda instead of pip (2.2.3 version will not be
available, which affects the lematizer behaviour)

To check if install worked do

```bash
. set_environment.sh
python tests/correctly_installed.py
```

As a further check, you can do a mini test with 25 annotated sentences that we
provide under DATA/, you can use this

```bash
bash tests/minimal_test.sh
```

This runs a full train test excluding alignment and should take around a
minute. Note that the model will not be able to learn from only 25 sentences.

The AMR aligner uses additional tools that can be donwloaded and installed with

```
bash preprocess/install_alignment_tools.sh
```

## Training a model

You first need to preprocess and align the data. For AMR2.0 do

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR/corpora/amr2.0/
```

The same for AMR1.0

```
python preprocess/merge_files.py /path/to/LDC2014T12/data/amrs/split/ DATA/AMR/corpora/amr1.0/
```

You will also need to unzip the precomputed BLINK cache

```
unzip /dccstor/ykt-parse/SHARED/CORPORA/EL/linkcache.zip
```

Then just call a config to carry a desired experiment

```bash
bash scripts/stack-transformer/experiment.sh configs/amr2_o5+Word100_roberta.large.top24_stnp6x6.sh
```

To display the results use

```bash
python scripts/stack-transformer/rank_results.py --seed-average
```

Note that there is cluster version of this script, currently only supporting
LSF but easily adaptable to e.g. Slurm

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse \
  --in-checkpoint $in_checkpoint \
  --in-tokenized-sentences $input_file \
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
