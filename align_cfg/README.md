# Install

```
conda create --name torch-1.2 --clone pytorch_1.2.0_py3.6_x86_64_v1
conda activate torch-1.2
cd transition-amr-parser
pip install -e .
pip install allennlp==1.0.0 -c align_cfg/constraints.txt
```

