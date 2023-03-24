# This script creates and optionally runs an experiment sweep for training and eval the aligner.

import argparse
import collections
import copy
import json
import random
import os
import time

from templates import template, eval_template, eval_dist_template

parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
parser.add_argument('--launch-eval', action='store_true')
args = parser.parse_args()

prefix = '2021-12-08a'

def default_model_cfg():
    cfg = {}
    cfg['text_emb'] = 'char'
    cfg['text_enc'] = 'bilstm'
    cfg['text_project'] = 200
    cfg['amr_emb'] = 'char'
    cfg['amr_enc'] = 'lstm' #
    cfg['amr_project'] = 200
    cfg['dropout'] = 0.3 #
    cfg['context'] = 'xy'
    cfg['hidden_size'] = 200
    cfg['prior'] = 'attn'
    cfg['output_mode'] = 'tied'
    cfg['num_amr_layers'] = 2
    return cfg

def default_flags():
    flags = {}
    flags['save-every-epoch'] = 10
    flags['lr'] = 2e-3
    flags['mask'] = 0 #
    flags['batch-size'] = 16
    flags['accum-steps'] = 8
    flags['max-epoch'] = 200
    return flags

def render_flags(flags):
    flags_str = ''

    for k, v in flags.items():
        if isinstance(k, bool):
            if v:
                flags_str += ' --{}'.format(v)

        else:
            flags_str += ' --{} {}'.format(k, v)

    return flags_str

def main():
    exp_list = []

    num_seeds = 3

    c_exp = collections.Counter()

    def arch_list():
        #for arch in ['attnC', 'lstmC', 'attnD', 'lstmD']:
        for arch in ['lstmC', 'lstmD']:
            for arch_size in [200, 400]:
                for accum_steps in [4, 8]:
                    yield arch, arch_size, accum_steps

    for i in range(num_seeds):
        for task in ['amr2']:
            for arch, arch_size, accum_steps in arch_list():
                # lrA - bad
                for lr_ in ['lrB', 'lrD']:
                    for mask_ in ['maskA']:
                        # dropC - bad
                        for drop_ in ['dropB']:
                            sofar = len(exp_list)

                            if sofar % 10 <= 6 or True:
                                partition = '1080ti-long'
                            else:
                                partition = '2080ti-long'

                            lr = {'lrA': 2e-3, 'lrB': 1e-4, 'lrC': 2e-4, 'lrD': 8e-5}[lr_]
                            mask = {'maskA': 0, 'maskB': 0.15}[mask_]
                            dropout = {'dropA': 0.1, 'dropB': 0.3, 'dropC': 0, 'dropD': 0.5}[drop_]

                            flags = default_flags()
                            flags['max-epoch'] = 200
                            flags['accum-steps'] = accum_steps
                            flags['lr'] = lr
                            flags['mask'] = mask
                            flags_str = render_flags(flags)

                            model_cfg = default_model_cfg()
                            if arch == 'lstmA':
                                model_cfg['text_enc'] = 'bilstm'
                                model_cfg['text_enc_cfg'] = dict(nlayers=1)
                                model_cfg['amr_enc'] = 'lstm'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=1)
                            elif arch == 'lstmB':
                                model_cfg['text_enc'] = 'bilstm'
                                model_cfg['text_enc_cfg'] = dict(nlayers=2)
                                model_cfg['amr_enc'] = 'lstm'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=2)
                            elif arch == 'lstmC':
                                model_cfg['text_enc'] = 'bilstm'
                                model_cfg['text_enc_cfg'] = dict(nlayers=2)
                                model_cfg['amr_enc'] = 'lstm'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=1)
                            elif arch == 'lstmD':
                                model_cfg['text_enc'] = 'bilstm'
                                model_cfg['text_enc_cfg'] = dict(nlayers=1)
                                model_cfg['amr_enc'] = 'lstm'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=2)
                            elif arch == 'attnA':
                                model_cfg['text_enc'] = 'bitransformer'
                                model_cfg['text_enc_cfg'] = dict(nlayers=2, nhead=2)
                                model_cfg['amr_enc'] = 'transformer'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=2, nhead=2)
                            elif arch == 'attnB':
                                model_cfg['text_enc'] = 'bitransformer'
                                model_cfg['text_enc_cfg'] = dict(nlayers=4, nhead=4)
                                model_cfg['amr_enc'] = 'transformer'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=4, nhead=4)
                            elif arch == 'attnC':
                                model_cfg['text_enc'] = 'bitransformer'
                                model_cfg['text_enc_cfg'] = dict(nlayers=3, nhead=2)
                                model_cfg['amr_enc'] = 'transformer'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=2, nhead=2)
                            elif arch == 'attnD':
                                model_cfg['text_enc'] = 'bitransformer'
                                model_cfg['text_enc_cfg'] = dict(nlayers=2, nhead=2)
                                model_cfg['amr_enc'] = 'transformer'
                                model_cfg['amr_enc_cfg'] = dict(nlayers=3, nhead=2)
                            model_cfg['hidden_size'] = arch_size
                            model_cfg['text_project'] = model_cfg['hidden_size']
                            model_cfg['amr_project'] = model_cfg['hidden_size']
                            model_cfg['dropout'] = dropout
                            model_cfg_str = " --model-config '{}'".format(json.dumps(model_cfg))

                            exp_key = '{}.{}.{}.{}.{}.{}'.format(task, arch, arch_size, lr_, drop_, mask_)
                            exp_id = c_exp[exp_key]
                            c_exp[exp_key] += 1

                            d = {}
                            d['exp_id'] = exp_id
                            d['exp_key'] = exp_key
                            d['prefix'] = prefix
                            d['partition'] = partition
                            d['task'] = task
                            d['flags'] = flags_str
                            d['model_cfg'] = model_cfg_str

                            # TRAIN
                            name = 'gypsum.{prefix}.{exp_key}.{exp_id}'.format(**d)
                            d['name'] = name

                            os.system('mkdir -p log/{}'.format(name))

                            script = template.format(**d)

                            path = 'log/{}/script.sh'.format(name)

                            with open(path, 'w') as f:
                                f.write(script)

                            ex = {}
                            ex['name'] = name
                            ex['script'] = script
                            ex['script_path'] = path
                            ex['log_path'] = 'log/{}'.format(name)
                            exp_list.append(ex)

                            print(name)
                            #print(script)

                            # EVAL

                            if sofar % 3 <= 1 or True:
                                partition = '1080ti-short'
                            else:
                                partition = '2080ti-short'

                            ex['eval_info'] = []

                            base_d = d
                            base_name = name

                            for i_exp in range(2):

                                for epoch in range(10, flags['max-epoch'], 10):
                                    model_epoch = 'epoch_{}'.format(epoch)

                                    d = copy.deepcopy(base_d)
                                    d['i_exp'] = i_exp

                                    if i_exp == 1:
                                        if not 'gcn' in ex['name']:
                                            continue
                                        d['flags'] = d['flags'] + ' --mask-at-inference'

                                    d['partition'] = partition
                                    d['model_epoch'] = model_epoch
                                    d['load'] = './log/{}/model.{}.pt'.format(base_name, model_epoch)
                                    name = 'gypsum.{prefix}.{exp_key}.{exp_id}.{i_exp}.eval.{model_epoch}'.format(**d)
                                    d['name'] = name

                                    os.system('mkdir -p log/{}'.format(name))
                                    script = eval_template.format(**d)
                                    path = 'log/{}/script.sh'.format(name)
                                    with open(path, 'w') as f:
                                        f.write(script)

                                    info = {}
                                    info['script'] = script
                                    info['script_path'] = path
                                    info['log_path'] = './log/{}'.format(name)
                                    info['load'] = d['load']
                                    info['epoch'] = epoch
                                    info['base_name'] = base_name
                                    ex['eval_info'].append(info)

                                    #print('eval', path)

                                    # write align dist
                                    d = copy.deepcopy(d)
                                    name = 'gypsum.{prefix}.{exp_key}.{exp_id}.{i_exp}.eval_align_dist.{model_epoch}'.format(**d)
                                    d['name'] = name

                                    os.system('mkdir -p log/{}'.format(name))
                                    script = eval_dist_template.format(**d)
                                    path = 'log/{}/script.sh'.format(name)
                                    with open(path, 'w') as f:
                                        f.write(script)

                                    info['align_dist_script_path'] = path

    file_list_path = 'eval_json.{}.txt'.format(prefix)
    with open(file_list_path, 'w') as f:
        for exp in exp_list:
            for info in exp['eval_info']:
                f.write('{} {}\n'.format(info['log_path'], exp['log_path']))


    if args.launch:
        exp_list_ = copy.deepcopy(exp_list)
        random.shuffle(exp_list_)
        for exp in exp_list_:
            os.system('sbatch {}'.format(exp['script_path']))


    if args.launch_eval:
        errors = collections.defaultdict(collections.Counter)

        to_run = []
        for exp in exp_list[::-1]:
            for info in exp['eval_info']:
                eval_json = os.path.join(info['log_path'], 'train.aligned.txt.eval.json')
                if os.path.exists(eval_json):
                    errors[info['base_name']]['skip'] += 1
                    continue
                if not os.path.exists(info['load']):
                    errors[info['base_name']]['no_ckpt'] += 1
                    continue

                to_run.append(info)

        for k, v in errors.items():
            print(k, v)

        sofar = 0
        sleep_amount = 10 * 60 // 2
        sleep_every = 50

        for info in sorted(to_run, key=lambda x: (x['epoch'], x['base_name'])):
            if True:
                path = info['script_path']
                os.system('sbatch {}'.format(path))
                sofar += 1

                if sofar % sleep_every == 0:
                    print('Sleeping for {} sec every {} jobs. {} jobs so far.'.format(sleep_amount, sleep_every, sofar))
                    time.sleep(sleep_amount)

    print(len(exp_list))

    print(file_list_path, sum([len(exp['eval_info']) for exp in exp_list]))


main()


