import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
parser.add_argument('--log', default='/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align')
parser.add_argument('--input', default='align_cfg/experiments.jsonl')
parser.add_argument('--require', default=None, type=str)
parser.add_argument('--queue', default='x86_24h', type=str)
parser.add_argument('--new', action='store_true')
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

queue = args.queue
require = '-require {}'.format(args.require) if args.require is not None else ''
conda = 'torch-1.2-new' if args.new else 'torch-1.2'

template = """#!/usr/bin/env bash

source /u/adrozdov/.bashrc

conda activate {conda}

cd /u/adrozdov/code/transition-amr-parser

python -u align_cfg/main.py --cuda \
    --log-dir {log} \
    {flags} \
    --jbsub-eval
"""

eval_template = """#!/usr/bin/env bash

source /u/adrozdov/.bashrc

conda activate {conda}

cd /u/adrozdov/code/transition-amr-parser

python -u align_cfg/main.py --cuda \
    --log-dir {log}_write_amr2 \
    {flags} \
    --load {log}/model.best.val_1_recall.pt \
    --trn-amr ~/data/AMR2.0/aligned/cofill/train.txt \
    --write-only \
    --batch-size 8 \
    --max-length 0

python -u align_cfg/main.py --cuda \
    --log-dir {log}_write_amr3 \
    {flags} \
    --load {log}/model.best.val_1_recall.pt \
    --trn-amr ~/data/AMR3.0/train.txt \
    --write-only \
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
        flags=flags)


def launch_exp(logdir, train=True):

    if train:
        stdout_path = os.path.join(logdir, 'stdout.txt')
        stderr_path = os.path.join(logdir, 'stderr.txt')
        script_path = os.path.join(logdir, 'script.txt')
    else:
        stdout_path = os.path.join(logdir, 'eval.stdout.txt')
        stderr_path = os.path.join(logdir, 'eval.stderr.txt')
        script_path = os.path.join(logdir, 'eval_script.txt')

    cmd = 'jbsub -cores 1+1  -q {queue} -out {stdout} -err {stderr} -mem 30g {require} bash {script} > tmp.out'.format(
            queue=queue,
            stdout=stdout_path,
            stderr=stderr_path,
            require=require,
            script=script_path)
    os.system(cmd)
    os.system('cat tmp.out')

    with open('tmp.out') as f:
        data = f.read()

    jobid = data.split('<')[-2].split('>')[0]

    return jobid


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

        if args.launch:
            jobid = launch_exp(logdir, train=True)

        # Auto-launch eval.
        script = render(eval_template, cfg, logdir)
        script_path = os.path.join(logdir, 'eval_script.txt')
        with open(script_path, 'w') as fs:
            fs.write(script)

        # Note: Does not appear to use -depend correctly.
        if args.launch and False:
            jobid = launch_exp(logdir, train=False)

