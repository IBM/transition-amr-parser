Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch version `0.3.4`. Current code implements the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. This yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## Installation

Clone and pip install (see `set_environment.sh` below if you use a virtualenv)

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
. set_environment.sh     # see below
pip install -r scripts/stack-transformer/requirements.txt
bash scripts/download_and_patch_fairseq.sh
pip install --no-deps --editable fairseq-stack-transformer
pip install .            # use --editable if to modify code
```

The code needs Pytorch `1.1.0` and Python `3.6-3.7`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

```bash
# inside set_environment.sh
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```
OR you can leave this empty and handle environment activation yourself i.e.

```bash
touch set_environment.sh
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

You will also need to unzip the precomputed BLINK cache (contact us to get this)

```
unzip /path/to/linkcache.zip
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
