Transition-based Neural Parser
============================


Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR parsing and reliable token to node alignments. 

### Pip Installation Instructions
1. After your pip install the package, you still need to install an additional dependency, torch-scatter.
If you are using Linux server; otherwise, please visit torch-scatter's official repo to find matching versions. 

```
conda create -n amr-parser python=3.8
conda activate amr-parser
pip install transition-neural-parser
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

2. Download and run inference using a pretrained AMR parser:
```
from transition_amr_parser.parse import AMRParser

# download and save to cache a model named AMR3.0
parser = AMRParser.from_pretrained('AMR3-structbart-L')
tokens, positions = parser.tokenize('The girl travels and visits places')

# use parse_sentences() for a batch of sentences
annotations, machines = parser.parse_sentence(tokens)

# Print Penman 
print(annotations)

# print into Penman w/o JAMR, ISI
amr = machines.get_amr()
print(amr.to_penman(jamr=False, isi=True))

# graph plot (needs matplotlib)
amr.plot()

```

### Available Pretrained Models
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

### Future Releases
The current release only supports model inference using python script; 
We will add training and evaluation script in future versions; interested users may refer to this repo for training and evaluation: [here](https://github.com/IBM/transition-amr-parser)

Additionally, Linux operating system and Python 3.8 are required environment now; future releases will also support MacOS and higher versions of python. 

Current and past releases:
v1.0.0 correspond to v0.5.2 in [here](https://github.com/IBM/transition-amr-parser).

### Structured-BART 

Current version (`0.5.2`). Structured-BART [(Zhou et al 2021b)](https://aclanthology.org/2021.emnlp-main.507/) encodes the parser state using specialized cross and self-attention heads and leverages BART's language model to replace the use of subgraph actions and lemmatizer, thus enabling a much simpler oracle with 100% coverage. It yields `84.2` Smatch (`84.7` with silver data and `84.9` with ensemble). Version `0.5.2` introduces the ibm-neural-aligner [(Drozdov et al 2022)](https://arxiv.org/abs/2205.01464) yielding a base AMR3.0 performance of `82.7` (`83.1` with latent alignment training). Structured-BART is also used for [(Lee et al 2022)](https://arxiv.org/abs/2112.07790) which yields a new single model SoTA of `85.7` for AMR2.0 and `84.1` for AMR3.0 by introducing Smatch-based ensemble distillation.

### Action Pointer

Checkout the `action-pointer` branch (derived from version `0.4.2`) for the `Action Pointer Transformer` model [(Zhou et al 2021)](https://www.aclweb.org/anthology/2021.naacl-main.443) from NAACL2021. As the stack-Transformer, APT encodes the parser state in dedicated attention heads. APT uses however actions creating nodes to represent them. This decouples token and node representations yielding much shorter sequences than previous oracles with higher coverage. APT achieves `81.8` Smatch (`83.4` with silver data and partial ensemble) on AMR2.0 test using RoBERTa embeddings and has an efficient shallow decoder. Due to aligner implementation improvements this code reaches `82.1` on AMR2.0 test, better that what is reported in the paper.

### Stack-Transformer

Checkout the `stack-transformer` branch (derived from version `0.3.4`) for the `stack-Transformer` model [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89) from EMNLP findings 2020. The stack-Transformer masks dedicated cross attention heads to encode the parser state represented by stack and buffer. It yields `80.2` Smatch (`81.3` with self-learning) on AMR2.0 test (this code reaches `80.5` due to the aligner implementation). Stack-Transformer can be used to reproduce our works on self-learning and cycle consistency in AMR parsing [(Lee et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.288/) from EMNLP findings 2020, alignment-based multi-lingual AMR parsing [(Sheth et al 2021)](https://www.aclweb.org/anthology/2021.eacl-main.30/) from EACL 2021 and Knowledge Base Question Answering [(Kapanipathi et al 2021)](https://arxiv.org/abs/2012.01707) from ACL findings 2021.

The code also contains an implementation of the AMR aligner from [(Naseem et al 2019)](https://www.aclweb.org/anthology/P19-1451/) with the forced-alignment introduced in [(Fernandez Astudillo et al 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.89).

Aside from listed [contributors](https://github.com/IBM/transition-amr-parser/graphs/contributors), the initial commit was developed by Miguel Ballesteros and Austin Blodgett while at IBM.


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
