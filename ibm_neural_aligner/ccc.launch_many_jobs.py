import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--log', default='/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align')
parser.add_argument('--input', default='ibm_neural_aligner/experiments.jsonl')
parser.add_argument('--require', default=None, type=str)
parser.add_argument('--queue', default='x86_24h', type=str)
parser.add_argument('--new', action='store_true')
parser.add_argument('--latest', action='store_true')
parser.add_argument('--force', action='store_true')
parser.add_argument('--jbsub-eval', action='store_true')
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--eval-pretty-only', action='store_true')
args = parser.parse_args()

queue = args.queue
require = '-require {}'.format(args.require) if args.require is not None else ''
conda = 'torch-1.4-new'
if args.latest:
    conda = '/dccstor/ykt-parse/SHARED/misc/adrozdov/envs/latest'

jbsub_eval = '--jbsub-eval' if args.jbsub_eval else ''


# Note: We use a sample of train for dev.

template = """#!/usr/bin/env bash

source /u/adrozdov/.bashrc

conda activate {conda}

cd /dccstor/ykt-parse/SHARED/misc/adrozdov/code/mnlp-transition-amr-parser

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u ibm_neural_aligner/main.py --cuda \
    --aligner-training-and-eval \
    --log-dir {log} \
    --max-length 100 \
    --trn-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.train-v1 \
    --val-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
    --cache-dir ./tmp-aligner \
    --vocab-text ./tmp-aligner/vocab.text.txt \
    --vocab-amr  ./tmp-aligner/vocab.amr.txt \
    {flags} \
    {jbsub_eval}
"""

eval_template = """#!/usr/bin/env bash

source /u/adrozdov/.bashrc

conda activate {conda}

cd /dccstor/ykt-parse/SHARED/misc/adrozdov/code/mnlp-transition-amr-parser

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u ibm_neural_aligner/main.py --cuda \
    --aligner-training-and-eval \
    --log-dir {log}_write_amr2 \
    {flags} \
    --load {log}/model.best.val_0_recall.pt \
    --trn-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt \
    --val-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
    --cache-dir ./tmp-aligner \
    --vocab-text ./tmp-aligner/vocab.text.txt \
    --vocab-amr  ./tmp-aligner/vocab.amr.txt \
    --write-only \
    --batch-size 8 \
    --max-length 0

python -u ibm_neural_aligner/main.py --cuda \
    --aligner-training-and-eval \
    --log-dir {log}_write_amr3 \
    {flags} \
    --load {log}/model.best.val_0_recall.pt \
    --trn-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/aligned/cofill/train.txt \
    --val-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
    --cache-dir ./tmp-aligner \
    --vocab-text ./tmp-aligner/vocab.text.txt \
    --vocab-amr  ./tmp-aligner/vocab.amr.txt \
    --write-only \
    --batch-size 8 \
    --max-length 0
"""

eval_pretty_template = """#!/usr/bin/env bash

source /u/adrozdov/.bashrc

conda activate {conda}

cd /dccstor/ykt-parse/SHARED/misc/adrozdov/code/mnlp-transition-amr-parser

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u ibm_neural_aligner/main.py --cuda \
    --aligner-training-and-eval \
    --log-dir {log}_write_pretty_amr2 \
    {flags} \
    --load {log}/model.best.val_0_recall.pt \
    --trn-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt \
    --val-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
    --cache-dir ./tmp-aligner \
    --vocab-text ./tmp-aligner/vocab.text.txt \
    --vocab-amr  ./tmp-aligner/vocab.amr.txt \
    --write-pretty \
    --batch-size 8 \
    --max-length 0

python -u ibm_neural_aligner/main.py --cuda \
    --aligner-training-and-eval \
    --log-dir {log}_write_pretty_amr3 \
    {flags} \
    --load {log}/model.best.val_0_recall.pt \
    --trn-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR3.0/aligned/cofill/train.txt \
    --val-amr /dccstor/ykt-parse/SHARED/misc/adrozdov/data/AMR2.0/aligned/cofill/train.txt.dev-seen-v1 \
    --cache-dir ./tmp-aligner \
    --vocab-text ./tmp-aligner/vocab.text.txt \
    --vocab-amr  ./tmp-aligner/vocab.amr.txt \
    --write-pretty \
    --batch-size 8 \
    --max-length 0
"""

def render(template, cfg, logdir):
    flags = ''

    for k, v in cfg.items():
        if isinstance(v, dict):
            flags += " --{} '{}'".format(k, json.dumps(v))
        elif isinstance(v, bool):
            if v:
                flags += " --{}".format(k)
        else:
            flags += " --{} {}".format(k, v)

    return template.format(
        conda=conda,
        log=logdir,
        flags=flags,
        jbsub_eval=jbsub_eval,
        )


def launch_exp(logdir, train=True, pretty=False, launch=False, verbose=False):

    if train:
        stdout_path = os.path.join(logdir, 'stdout.txt')
        stderr_path = os.path.join(logdir, 'stderr.txt')
        script_path = os.path.join(logdir, 'script.txt')
    else:
        stdout_path = os.path.join(logdir, 'eval.stdout.txt')
        stderr_path = os.path.join(logdir, 'eval.stderr.txt')
        script_path = os.path.join(logdir, 'eval_script.txt')
    if pretty:
        stdout_path = os.path.join(logdir, 'eval_pretty.stdout.txt')
        stderr_path = os.path.join(logdir, 'eval_pretty.stderr.txt')
        script_path = os.path.join(logdir, 'eval_pretty.script.txt')


    cmd = 'jbsub -cores 1+1  -q {queue} -out {stdout} -err {stderr} -mem 30g {require} bash {script} > tmp.out'.format(
            queue=queue,
            stdout=stdout_path,
            stderr=stderr_path,
            require=require,
            script=script_path)

    if verbose:
        print(cmd)

    if launch:
        os.system(cmd)
        os.system('cat tmp.out')

        with open('tmp.out') as f:
            data = f.read()

        jobid = data.split('<')[-2].split('>')[0]

        return jobid

    return None


with open(args.input) as f:
    for line in f:
        if not line.strip() or line.startswith('#'):
            continue
        cfg = json.loads(line)
        name = cfg['name']

        logdir = os.path.join(args.log, name)
        if not args.force and os.path.exists(logdir):
            raise Exception('Already exists. Use --force. {}'.format(logdir))
        os.system('mkdir -p {}'.format(logdir))
        print(logdir)

        script = render(template, cfg, logdir)
        script_path = os.path.join(logdir, 'script.txt')
        with open(script_path, 'w') as fs:
            fs.write(script)

        # Auto-launch eval.
        script = render(eval_template, cfg, logdir)
        script_path = os.path.join(logdir, 'eval_script.txt')
        with open(script_path, 'w') as fs:
            fs.write(script)

        script = render(eval_pretty_template, cfg, logdir)
        script_path = os.path.join(logdir, 'eval_pretty.script.txt')
        with open(script_path, 'w') as fs:
            fs.write(script)

        if args.eval_only:
            jobid = launch_exp(logdir, train=False, launch=args.launch, verbose=args.verbose)
            continue
        if args.eval_pretty_only:
            jobid = launch_exp(logdir, train=False, pretty=True, launch=args.launch, verbose=args.verbose)
            continue

        if args.launch:
            jobid = launch_exp(logdir, train=True, launch=True)

