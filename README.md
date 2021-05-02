Transition-based AMR Parser
============================

Ongoing APT-O10-BART model

Fast install on CCC (and maybe on x86 machines in general)

```
git clone git@github.com:jzhou316/transition-amr-parser.git
cd transition-amr-parser
git checkout apt-bart-o10
# Activate your virtualenv inside this file, for example write
# eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
# [ ! -d cenv_x86 ] && conda create -y -p ./cenv_x86
# conda activate ./cenv_x86
# or else empty file
touch set_environment.sh
bash scripts/install_x86_with_conda.sh
```

Fasts tests, install went ok

```
bash tests/correctly_installed.sh
```

Fast full train/test with 25 sentences

```
bash tests/minimal_test.sh
```

Expected data structure

```
# amr
amr_corpus/amr2.0/o5/jkaln.txt
amr_corpus/amr2.0/o5/dev.txt.removedWiki.noempty.JAMRaligned
amr_corpus/amr2.0/o5/test.txt.removedWiki.noempty.JAMRaligned

# wiki
amr_corpus/amr2.0/wiki/dev.wiki
amr_corpus/amr2.0/wiki/dev.txt
amr_corpus/amr2.0/wiki/test.wiki
amr_corpus/amr2.0/wiki/test.txt
```

## Running Scripts

First, make sure in the data configuration files (stored in config_files/config_data), the paths to data are correct.

Make sure you have the model configuartion file (usually stored in config_files/) you want to run.

Make a folder to record experiment launch time and pid: `mkdir .jbsub_logs`

1. run the data pre-processing:
```
bash run_tp/run_data.sh [data_config_file]
```

2. run the model (data pre-processing is included) **without** epoch evaluation:
```
CUDA_VISIBLE_DEVICES=0 bash run_tp/run_model_action-pointer.sh [model_config_file] [seed (optional); default is 42]
```

3. run **both** the model (data pre-processing is included) **and** the epoch evaluation:
```
CUDA_VISIBLE_DEVICES=0 bash run_tp/jbash_run_model-eval.sh [model_config_file] [seed (optional); default is 42]
```
this would run the two separate processes (model training and epoch evaluation) on the same GPU. To run them on different GPUs, one could use
```
bash run_tp/jbash_run_model-eval_2gpus.sh [model_config_file] [seed] [gpu_id_for_training] [gpu_id_for_evaluation]
```

4. run **only** the evaluation (epoch beam 1 valid evaluation and model selection, averaging, and final testing):
```
CUDA_VISIBLE_DEVICES=0 bash run_tp/jbash_run_eval.sh [model_config_file] [seed (optional); default is 42]
```

## Running Scripts on CCC

First, make sure in the data configuration files (stored in config_files/config_data), the paths to data are correct.

Make sure you have the model configuartion file (usually stored in config_files/) you want to run.

Make a folder to record experiment launch time and pid: `mkdir .jbsub_logs`

1. run the data pre-processing:

submit the following interactive bash command to CCC
```
bash run_tp/run_data.sh [data_config_file]
```

2. run the model (data pre-processing is included) **without** epoch evaluation:

submit the corresponding interactive bash command (above) to CCC, or use the following script directly where job submission is wrapped inside
```
bash run_tp/jbsub_run_model.sh [model_config_file] [seed (optional); default is 42]
```

3. run **both** the model (data pre-processing is included) **and** the epoch evaluation:

submit the corresponding interactive bash command (above) to CCC, or use the following script directly where job submission is wrapped inside
```
bash run_tp/jbsub_run_model-eval.sh [model_config_file] [seed (optional); default is 42]
```
this would run the two separate processes (model training and epoch evaluation) on the same GPU.

4. run **only** the evaluation (epoch beam 1 valid evaluation and model selection, averaging, and final testing):

submit the corresponding interactive bash command (above) to CCC, or use the following script directly where job submission is wrapped inside
```
bash run_tp/jbsub_run_eval.sh [model_config_file] [seed (optional); default is 42]
```

5. run 3 random seeds, both the model (data pre-processing is included) and the epoch evaluation:

submit 3 jobs separately, or use the following script directly where job submission is wrapped inside
```
bash run_tp/jbsub_run_model-eval_seeds.sh [model_config_file]
```

## Collect Model Scores

To output a summary of model scores:
```
bash run_tp/collect_scores.sh [checkpoint_folder]
```
e.g. the `[checkpoint_folder]` is of form `EXP/exp_[data&model_config_tags]/models_ep120_seed42_[optimization_tags]/` after training and evaluation is done.

To examine the epoch testing and model selection results (ranked smatch scores, beam 1 on dev data):
```
less [checkpoint_folder]/epoch_wiki-smatch_ranks.txt
```
