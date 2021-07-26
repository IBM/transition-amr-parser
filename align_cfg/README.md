# Install

```
cd transition-amr-parser

conda create --name torch-1.4 python=3.6
conda activate torch-1.4
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install -e .
conda install -c dglteam "dgl-cuda10.1<0.5"
```

Changes for CPU:

```
conda install -y pytorch==1.4.0 torchvision==0.5.0 -c pytorch
pip install dgl==0.4.3.post2
```
