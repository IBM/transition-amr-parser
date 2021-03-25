Transition-based AMR Parser
============================

Transition-based parser for Abstract Meaning Representation (AMR) in Pytorch. Current version (`v0.4.0`) implements the `action-pointer` model [(Zhou et al 2021)](https://openreview.net/forum?id=X9KK-SCmKWn). For the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://arxiv.org/abs/2010.10669) checkout `v0.3.3`. Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

Check [Parsing Services](https://github.ibm.com/mnlp/transition-amr-parser/wiki/Parsing-Services) for the endpoint URLs and Docker instructions. If you have acess to CCC and LDC data, we have available both the train data and trained models.

## Installation

We use a `set_environment.sh` script to activate conda/pyenv and virtual
environments. You can leave this empty if you dont want to use it, but scripts
will assume at least an empty file exists.
```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
touch set_environment.sh
. set_environment.sh
pip install .
```

The AMR aligner uses additional tools that can be donwloaded and installed with

```
bash preprocess/install_alignment_tools.sh
```

If you use already aligned AMR, you will not need this.

## Installation Details

An example of `set_environment.sh`
```
# Activate conda and local virtualenv for this machine
eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
[ ! -d cenv_x86 ] && conda create -y -p ./cenv_x86
conda activate ./cenv_x86
```

The code has been tested on Python `3.6` and `3.7` (x86 only). Alternatively,
you may pre-install some of the packages with conda, if this works better on
your achitecture, and the do the pip install above. You will need this for PPC
instals.
```
conda install pytorch=1.3.0 -y -c pytorch
conda install -c conda-forge nvidia-apex -y
```

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 1-3 minutes. It wont learn anything but at least will run all stages.
```bash
bash tests/minimal_test.sh
```

## Training a model

You first need to preprocess and align the data. For AMR2.0 do

```bash
. set_environment.sh
python preprocess/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

The same for AMR1.0

```
python preprocess/merge_files.py /path/to/LDC2014T12/data/amrs/split/ DATA/AMR2.0/corpora/
```

You will also need to unzip the precomputed BLINK cache

```
unzip /dccstor/ykt-parse/SHARED/CORPORA/EL/linkcache.zip
```

To launch train/test use

```
bash run/run_experiment.sh configs/amr2.0-action-pointer.sh
```

you can check training status with

```
python run/status.py --config configs/amr2.0-action-pointer.sh
```

Note that for CCC there is a version using `jbsub` that split the task into
multiple sequential jobs and supports multiple seeds and testing in paralell

```
bash run/lsf/run_experiment.sh configs/amr2.0-action-pointer.sh
``` 

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
