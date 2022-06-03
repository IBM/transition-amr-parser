

template = """#!/bin/bash
#
#SBATCH --job-name={name}
#SBATCH -o /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser/log/{name}/slurm.out
#SBATCH --time=1-00:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45GB
#SBATCH --exclude=node030,node181,node108,node171,node158,node176,node105

CACHE='cache-{task}'
LOG2='./log/{name}'

source activate ibm-amr-aligner-torch-1.8-v2
cd /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser

date

python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --max-length 100 --log-dir $LOG2 --max-epoch 400  --batch-size 32 --accum-steps 4 --verbose  {flags} {model_cfg}
"""
# train
#python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --lr 2e-3 --max-length 100 --log-dir $LOG2 --max-epoch 200 --model-config '{{"text_emb": "char", "text_enc": "bilstm", "text_project": 200, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 200, "dropout": 0.3, "context": "xy", "hidden_size": 200, "prior": "attn", "output_mode": "tied"}}' --batch-size 32 --accum-steps 4 --verbose --skip-validation

eval_ref_template = """#!/bin/bash
#
#SBATCH --job-name={name}
#SBATCH -o /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser/log/{name}/slurm.out
#SBATCH --time=0-02:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15GB
#SBATCH --exclude=node030,node181,node108,node171,node158,node176,node105

CACHE='cache-{task}'
LOG2='./log/{name}'

source activate ibm-amr-aligner-torch-1.8-v2
cd /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser

date

python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --max-length -1 --log-dir $LOG2 --max-epoch 400 --batch-size 32 --accum-steps 4 --verbose --skip-validation --load {load} --write-single --single-input ./$CACHE/lstm_summer_2021.train.amr --single-output ./$LOG2/lstm_summer_2021.train.amr --aligner-training-and-eval {flags} {model_cfg}

python ibm_neural_aligner/run_eval.py --pred ./$LOG2/lstm_summer_2021.train.amr --gold ./$CACHE/lstm_summer_2021.train.amr
"""

eval_template = """#!/bin/bash
#
#SBATCH --job-name={name}
#SBATCH -o /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser/log/{name}/slurm.out
#SBATCH --time=0-02:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15GB
#SBATCH --exclude=node030,node181,node108,node171,node158,node176,node105

CACHE='cache-{task}'
LOG2='./log/{name}'

if [ -f $LOG2/train.aligned.txt.eval.json ]; then
    echo "Found eval output! $LOG2/train.aligned.txt.eval.json"
    exit 0
fi

source activate ibm-amr-aligner-torch-1.8-v2
cd /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser

date

python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/dev.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --max-length -1 --log-dir $LOG2 --max-epoch 400 --batch-size 32 --accum-steps 4 --verbose --skip-validation --load {load} --write-single --single-input ./$CACHE/dev.txt.no_wiki --single-output ./$LOG2/dev.aligned.txt {flags} {model_cfg}

python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --max-length -1 --log-dir $LOG2 --max-epoch 400 --batch-size 32 --accum-steps 4 --verbose --skip-validation --load {load} --write-single --single-input ./$CACHE/train.aligned.txt --single-output ./$LOG2/train.aligned.txt --aligner-training-and-eval {flags} {model_cfg}

python ibm_neural_aligner/run_eval.py --pred ./$LOG2/train.aligned.txt --gold ./$CACHE/train.aligned.txt
"""

eval_dist_template = """#!/bin/bash
#
#SBATCH --job-name={name}
#SBATCH -o /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser/log/{name}/slurm.out
#SBATCH --time=0-02:00:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15GB
#SBATCH --exclude=node030,node181,node108,node171,node158,node176,node105

CACHE='cache-{task}'
LOG2='./log/{name}'

source activate ibm-amr-aligner-torch-1.8-v2
cd /mnt/nfs/work1/mccallum/adrozdov/code/transition-amr-parser

date

python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --max-length -1 --log-dir $LOG2 --max-epoch 400 --batch-size 32 --accum-steps 4 --verbose --skip-validation --load {load} --write-align-dist --single-input ./$CACHE/train.dummy_align.txt --single-output ./$LOG2/train.txt.align_dist.npy {flags} {model_cfg}

python -u ibm_neural_aligner/align_utils.py write_argmax --ibm-format --in-amr $CACHE/train.dummy_align.txt --in-amr-align-dist ./$LOG2/train.txt.align_dist.npy --out-amr-aligned ./$LOG2/train.txt

python -u ibm_neural_aligner/align_utils.py verify_corpus_id --ibm-format --in-amr ./$LOG2/train.txt --corpus-id ./$LOG2/train.txt.align_dist.npy.corpus_hash

"""

# eval
#python -u ibm_neural_aligner/main.py --cuda --cache-dir $CACHE --vocab-text ./$CACHE/vocab.text.txt --vocab-amr ./$CACHE/vocab.amr.txt  --trn-amr ./$CACHE/train.txt.no_wiki --val-amr ./$CACHE/dev.txt.no_wiki --tst-amr ./$CACHE/test.txt.no_wiki --lr 2e-3 --max-length -1 --log-dir $LOG2 --max-epoch 200 --model-config '{{"text_emb": "char", "text_enc": "bilstm", "text_project": 200, "amr_emb": "char", "amr_enc": "lstm", "amr_project": 200, "dropout": 0.3, "context": "xy", "hidden_size": 200, "prior":"attn", "output_mode": "tied"}}' --batch-size 32 --accum-steps 4 --verbose --skip-validation --load {load} --write-single --single-input ./$CACHE/train.txt.no_wiki --single-output ./$LOG2/train.txt.no_wiki
