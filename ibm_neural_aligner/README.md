# Install (Compatible w. AMR Parser)

```
cd transition-amr-parser

conda create --name torch-1.4 python=3.6
conda activate torch-1.4
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install h5py # Required for elmo embeddings.
pip install -e .
conda install -c dglteam "dgl-cuda10.1<0.5"
```

Changes for CPU:

```
conda install -y pytorch==1.4.0 torchvision==0.5.0 -c pytorch
pip install dgl==0.4.3.post2
```

For GCN support, need to install latest torch-geometric.

# Install (with newer torch for easy GCN support)

```
conda create -n ibm-amr-aligner python=3.8 -y
conda activate ibm-amr-aligner

# Use torch 1.8, since newer causes issue in torch-geometric.
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

pip install h5py # Required for elmo embeddings.
# (NOT TESTED) # conda install -c dglteam dgl-cuda11.1 -y # Installs DGL for TreeLSTM support.
conda install pyg -c pyg -c conda-forge -y # Installs torch-geometric for GCN support.

# The next step is tricky. Need to install AMR parser, but requires modifying `setup.py`

# Step 1:
vim setup.py # Comment out line about torch 1.4.

# Step 2:
pip install -e .
```

