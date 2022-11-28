Transition-based AMR Parser
============================

Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR parsing and reliable token to node alignments. See below for the different versions and corresponding papers. For trained checkpoints see [here](#trained-checkpoints).

- (✨New✨) [Smatch significance testing](scripts/README.md#paired-boostrap-significance-test-for-Smatch): Adds to the regular [Smatch](https://github.com/snowblink14/smatch) tool a significance test with almost no computation overhead. Can test multiple systems for pair-wise significance.

- (✨New✨) [Maximum Bayes Smatch Ensemble Distillation checkpoints](#trained-checkpoints): Includes the three seeds for the ensemble. These are SoTA for AMR parsing. 

### Structured-BART 

Current version (`0.5.2`). Structured-BART [(Zhou et al 2021b)](https://aclanthology.org/2021.emnlp-main.507/) encodes the parser state using specialized cross and self-attention heads and leverages BART's language model to replace the use of subgraph actions and lemmatizer, thus enabling a much simpler oracle with 100% coverage. It yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble). Version `0.5.2` introduces the ibm-neural-aligner [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) yielding a base AMR3.0 performance of `82.7` (`83.1` with latent alignment training). Structured-BART is also used for [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) which yields a new single model SoTA of `85.7` for AMR2.0 and `84.1` for AMR3.0 by introducing Smatch-based ensemble distillation.

### Action Pointer

Checkout the `action-pointer` branch (derived from version `0.4.2`) for the `Action Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. As the stack-Transformer, APT encodes the parser state in dedicated attention heads. APT uses however actions creating nodes to represent them. This decouples token and node representations yielding much shorter sequences than previous oracles with higher coverage. APT achieves `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test using RoBERTa embeddings and has an efficient shallow decoder. Due to aligner implementation improvements this code reaches `82.1` on AMR2.0 test, better that what is reported in the paper.

### Stack-Transformer

Checkout the `stack-transformer` branch (derived from version `0.3.4`) for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. The stack-Transformer masks dedicated cross attention heads to encode the parser state represented by stack and buffer. It yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.

## IBM Internal Features

IBM-ers please look [here](https://github.ibm.com/mnlp/transition-amr-parser/wiki) for available parsing web-services, CCC installers/trainers, trained models, etc. 

## Installation

The code needs Pytorch `1.10` and fairseq `0.10.2`. We tested it with Python `3.6-3.7`. We use a `set_environment.sh` script inside of which we activate conda/pyenv and virtual environments, it can contain for example 

```bash
# inside set_environment.sh
[ ! -d venv ] && virtualenv venv
. venv/bin/activate
```
OR you can leave this empty and handle environment activation yourself i.e.

```bash
touch set_environment.sh
```

Note that all bash scripts always source `set_environment.sh`, so you do not need to source it yourself.

To install clone and pip install 

```bash
git clone git@github.ibm.com:mnlp/transition-amr-parser.git
cd transition-amr-parser
git checkout <branch>     # for e.g. action-pointer, ignore for current version
. set_environment.sh      # see above
pip install --editable .   
```

it installs correctly both on OSX and Linux (RHEL). For linux it may be
easier to pre-install with conda targeting your architecture, for example

    conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

To test if install worked
```bash
bash tests/correctly_installed.sh
```
To do a mini-test with 25 annotated sentences that we provide. This should take 10 minutes. It wont learn anything but at least will run all stages.
```bash
bash tests/minimal_test.sh
```

## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
. set_environment.sh
python scripts/merge_files.py /path/to/LDC2017T10/data/amrs/split/ DATA/AMR2.0/corpora/
```

You will also need to unzip the precomputed BLINK cache. See issues in this repository to get the cache file (or the link above for IBM-ers).

```
unzip /path/to/linkcache.zip
```

To launch train/test use (this will also run the aligner)

```
bash run/run_experiment.sh configs/amr2.0-structured-bart-large.sh
```

Training will store and evaluate all checkpoints by default (see config's
`EVAL_INIT_EPOCH`) and select the one with best dev Smatch. This needs a lot of
space but you can launch a parallel job that will perform evaluation and delete
Checkpoints not in the top `5` 

```
bash run/run_model_eval.sh configs/amr2.0-structured-bart-large.sh
```

you can check training status with

```
python run/status.py -c configs/amr2.0-structured-bart-large.sh
```

use `--results` to check for scores once models are finished.

We include code to launch parallel jobs in the LSF job schedules. This can be
adapted for other schedulers e.g. Slurm, see [here](run/lsf/README.md)

## Decode with Pre-trained model

To use from the command line with a trained model do

```bash
amr-parse -c $in_checkpoint -i $input_file -o file.amr
```

It will parse each line of `$input_file` separately. It assumes tokenization,
use `--tokenize` otherwise. Once a model is unzipped, `-m <config>` can be used
instead of `-c`. The `file.amr` will contain the PENMAN notation with ISI
alignment annotations (`<node name>~<token position>`). Note that Smatch does
not support ISI and gives worse results. Use `--no-isi` to store alignments in
`::alignments` meta data. Also use `--jamr` to add JAMR annotations in
meta-data.

To use from other Python code with a trained model do

```python
from transition_amr_parser.parse import AMRParser
parser = AMRParser.from_checkpoint(checkpoint_path)
tokens, positions = parser.tokenize('The girl travels and visits places')
# use parse_sentences() for a batch of sentences
annotations, decoding_data = parser.parse_sentence(tokens)
# Print Penman 
print(annotations)
# transition_amr_parser.amr:AMR from transition_amr_parser.amr_machine:AMRStateMAchine
amr = decoding_data['machine'].get_amr()
# print into Penman w/o JAMR, ISI
print(amr.to_penman(jamr=False, isi=True))
# graph plot (needs matplotlib)
amr.plot()
```

## Trained checkpoints

We offer some trained checkpoints on demand. These can be download from AWS by using

    pip install awscli
    aws --endpoint-url=$URL s3 cp s3://mnlp-models-amr/<config>(-seed<N>).zip .
    unzip <config>(-seed<N>).zip

you will need access keys and URL. We provide these on an individual basis (sends us an email). For updates on available models see [here](https://twitter.com/RamonAstudill12). After unzipping, parsers should also be available by name from any folder as `AMRParser.load('<config>')`

Current available parsers are

|  paper                                                          |  config(.zip)                                         | beam    | Smatch  |
|:---------------------------------------------------------------:|:------------------------------------------------------:|:-------:|:-------:|
| [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) MAP    | amr2.0-structured-bart-large-neur-al-seed42            |   10    |   84.0  |
| [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) MAP    | amr3.0-structured-bart-large-neur-al-seed42            |   10    |   82.6  |
| [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) PR     | amr3.0-structured-bart-large-neur-al-sampling5-seed42  |   1     |   82.9  |
| [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | amr2joint_ontowiki2_g2g-structured-bart-large          |   10    |   85.9  |  
| [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) (ensemble) | amr3joint_ontowiki2_g2g-structured-bart-large          |   10    |   84.4  |  

we also provide the trained `ibm-neural-aligner` under names `AMR2.0_ibm_neural_aligner.zip` and `AMR3.0_ibm_neural_aligner.zip`. For the ensemble we provide the three seeds. Following fairseq conventions, to run the ensemble just give the three checkpoint paths joined by `:` to the normal checkpoint argument `-c`. Note that the checkpoints were trained with the `v0.5.1` tokenizer, this reduces performance by `0.1` on `v0.5.2` tokenized data.

Note that we allways report average of three seeds in papers while these are individual models. A fast way to test models standalone is

    bash tests/standalone.sh configs/<config>.sh
