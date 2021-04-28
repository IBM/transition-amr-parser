Transition-based AMR Parser
============================

Ongoing APT-O10-BART model

Fast install on CCC (and maybe on x86 machines in general)

```
git clone git@github.com:jzhou316/transition-amr-parser.git
cd transition-amr-parser
git checkout apt-bart-o10
# Activate your virtualenv inside this file, for example
# eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
# [ ! -d cenv_x86 ] && conda create -y -p ./cenv_x86
# conda activate ./cenv_x86
# or else empty file
touch set_environment.sh
bash scripts/install_x86_with_conda.sh
```

Fasts tests, install

```
bash tests/correctly_installed.sh
```

Full train test with 25 sentences

```
bash tests/minimal_test.sh
```
