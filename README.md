Transition-based Neural Parser
============================


## transition-neural-parser
**transition-neural-parser** is a powerful and easy-to-use Python package that provides a state-of-the-art neural transition-based parser for Abstract Meaning Representation (AMR). 

AMR is a semantic formalism used to represent the meaning of natural language sentences in a structured and machine-readable format. The package is designed to enable users to perform AMR parsing with high accuracy and generate reliable token-to-node alignments, which are crucial for various natural language understanding and generation tasks.


## Pip Installation Instructions
**Step 1: Create and activate a new conda environment;**

To ensure compatibility and prevent potential conflicts, create a new conda environment with Python 3.8:

```
conda create -y -p ./cenv_x86 python=3.8
```

Activate the newly created environment:

```
conda activate ./cenv_x86
```

**Step 2: Install the package**

Install the transition-neural-parser package using pip:

```
pip install transition-neural-parser
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

Alternatively, you can install locally; 
Go to the root directory of the project and then run the following:

```
pip install -e .
```

Note: The torch-scatter package is automatically set-up for most users using our default torch-1.13.1 and cu117 environment. However, in case you are choosing to use different environment set-up, or using non-linux servers, please visit the official [torch-scatter repository](https://pypi.org/project/torch-scatter/) to find the appropriate installation instructions.


**Step 3: Install docAMR repo**

To be able to run docamr , the docAMR repo must be cloned and installed.

Link to NAACL 2022 paper DOCAMR: Multi-Sentence AMR Representation and Evaluation
https://aclanthology.org/2022.naacl-main.256.pdf

```
git clone https://github.com/IBM/docAMR.git
cd docAMR
pip install .
cd ..

```


**(Optional) Step 4: Set a environment file (only for bash script training and evaluation)**

We use a set_environment.sh script inside of which we activate conda environment; it is used for model training.

First create the file at top-level directory of the project.
```
touch set_environment.sh
```

An example would be following, where the only line to change is where to source the conda.sh file. 
```
# inside set_environment.sh
if [ ! -d cenv_x86 ];then
    echo "Environment cenv_x86 not found"
    exit 1
else
    printf "\033[94mconda activate ./cenv_x86\033[0m\n"
    # replace the below line with the path your local conda location. 
    source /dccstor/gxamr/anaconda3/etc/profile.d/conda.sh.  # Please change this line
    conda activate ./cenv_x86
fi
```
Note that all bash scripts always source set_environment.sh, so you do not need to source it yourself anymore. 


## Decode with pretrained model
**Python Option:** Download a pretrained AMR parser and run inference;

Here is an example of how to download and use a pretrained AMR parser:

```
from transition_amr_parser.parse import AMRParser

# Download and save a model named AMR3.0 to cache
parser = AMRParser.from_pretrained('AMR3-structbart-L')
tokens, positions = parser.tokenize('The girl travels and visits places')

# Use parse_sentence() for single sentences or parse_sentences() for a batch
annotations, machines = parser.parse_sentence(tokens)

# Print Penman notation
print(annotations)

# Print Penman notation without JAMR, with ISI
amr = machines.get_amr()
print(amr.to_penman(jamr=False, isi=True))

# Plot the graph (requires matplotlib)
amr.plot()

```

This example demonstrates how to tokenize a document (list of sentences), parse it using the pretrained DocAMR parser, and print the resulting DocAMR graph in Penman notation. If you have matplotlib installed, you can also visualize the graph.
The resulting graph represents coreference using *:same-as* edges. To change the representation and merge the coreferent nodes as in the paper, please refer to [the DocAMR repo](https://github.com/IBM/docAMR.git)

*DocAMR*

```
from transition_amr_parser.parse import AMRParser

# Download and save the docamr model to cache
parser = AMRParser.from_pretrained('doc-sen-conll-amr-seed42')

# Sentences in the doc
doc = ["Hailey likes to travel." ,"She is going to London tomorrow.", "She will walk to Big Ben when she goes to London."]

# tokenize sentences if not already tokenized
tok_sentences = []
for sen in doc:
    tokens, positions = parser.tokenize(sen)
    tok_sentences.append(tokens)

# parse docs takes a list of docs as input
annotations, machines = parser.parse_docs([tok_sentences])

# Print Penman notation
print(annotations[0])

# Print Penman notation without JAMR, with ISI
amr = machines[0].get_amr()
print(amr.to_penman(jamr=False, isi=True))

# Plot the graph (requires matplotlib)
amr.plot()

```

**Command Line Option:** Use the command line to run a pretrained model to parse a file:

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

*DocAMR*

To parse a document , 
the input file ($doc_input_file) is a text file where each line is a sentence in the document and there is a newline ('\n') separating every doc (even at the end) 


```bash
amr-parse -c $in_checkpoint --in-doc $doc_input_file -o file.docamr --sliding
```

This will output a ".force_actions" file to the same directory as input , containing the actions needed to force sentence ends in the document as well as the output docamr "file.docamr"


## Available Pretrained Models
The models downloaded using from_pretrained() method will be stored to the pytorch cache folder as follows:
```
cache_dir = torch.hub._get_torch_home()
```

This table shows you available pretrained model names to download;

| pretrained model name                       | corresponding file name| 
|:----------------------------------------|:-----------:|
| AMR3-structbart-L-smpl                | amr3.0-structured-bart-large-neur-al-sampling5-seed42.zip      | 
| AMR3-structbart-L                     | amr3.0-structured-bart-large-neur-al-seed42.zip      | 
| AMR2-structbart-L                     | amr2.0-structured-bart-large-neur-al-seed42.zip      |
| AMR2-joint-ontowiki-seed42            | amr2joint_ontowiki2_g2g-structured-bart-large-seed42.zip       | 
| AMR2-joint-ontowiki-seed43            | amr2joint_ontowiki2_g2g-structured-bart-large-seed43.zip      | 
| AMR2-joint-ontowiki-seed44            | amr2joint_ontowiki2_g2g-structured-bart-large-seed44.zip      | 
| AMR3-joint-ontowiki-seed42            | amr3joint_ontowiki2_g2g-structured-bart-large-seed42.zip      | 
| AMR3-joint-ontowiki-seed43            | amr3joint_ontowiki2_g2g-structured-bart-large-seed43.zip      | 
| AMR3-joint-ontowiki-seed44            | amr3joint_ontowiki2_g2g-structured-bart-large-seed44.zip      | 
| doc-sen-conll-amr-seed42              |
both_doc+sen_trainsliding_ws400x100-seed42.zip                |


## Training a model

You first need to pre-process and align the data. For AMR2.0 do

```bash
conda activate ./cenv_x86 # activate parser environment
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


## Upcoming Features

The current release primarily supports model inference using Python scripts. In future versions, we plan to expand the capabilities of this package by:

- Adding python training and evaluation scripts for a more comprehensive user experience. Interested users can refer to the [IBM/transition-amr-parser](https://github.com/IBM/transition-amr-parser) repository for training and evaluation in the meantime.
- Broadening platform support to include M1 MacOS and higher versions of Python, in addition to the current support for the Linux operating system and Python 3.8.


## Research and Evaluation Results
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

## Evaluating Trained checkpoints

We offer some trained checkpoints on demand, and their evalution score measured in Smatch is below:

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
