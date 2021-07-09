import json
import itertools
import numpy as np


def product_dict(o):
    keys = o.keys()
    vals = o.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

version = '0.1'

seeds = 1

all_settings = []

def default_s():
    settings = {}
    settings['model-config.text_emb'] = ["char"]
    settings['model-config.text_enc'] = ["bilstm"]
    settings['model-config.text_project'] = [0]
    settings['model-config.amr_emb'] = ["char"]
    settings['model-config.amr_enc'] = ["lstm"]
    settings['model-config.amr_project'] = [0]
    settings['model-config.dropout'] = [0]
    settings['model-config.context'] = ['xy']
    settings['model-config.hidden_size'] = [100]
    settings['model-config.prior'] = ['attn']
    settings['batch-size'] = [4]
    settings['lr'] = [1e-3]
    settings['max-epoch'] = [20]
    return settings


settings = default_s()
settings['model-config.text_emb'] = ["char"]
settings['model-config.text_enc'] = ["bilstm"]
settings['model-config.text_project'] = [0]
settings['model-config.amr_emb'] = ["char"]
settings['model-config.amr_enc'] = ["tree_rnn"]
settings['model-config.amr_project'] = [0]
settings['model-config.dropout'] = [0, 0.1, 0.5]
settings['model-config.context'] = ['xy']
settings['model-config.hidden_size'] = [100, 200, 400]
settings['batch-size'] = [8]
settings['lr'] = [1e-3, 1e-4, 1e-5]
settings['max-epoch'] = [20]
all_settings.append(settings)

settings = default_s()
settings['model-config.text_emb'] = ["char"]
settings['model-config.text_enc'] = ["bilstm"]
settings['model-config.text_project'] = [0, 100]
settings['model-config.amr_emb'] = ["word", "char", "word+char"]
settings['model-config.amr_enc'] = ["lstm"]
settings['model-config.amr_project'] = [0, 100]
settings['model-config.context'] = ['xy']
settings['model-config.hidden_size'] = [100]
settings['batch-size'] = [4]
settings['lr'] = [1e-3]
settings['max-epoch'] = [20]
all_settings.append(settings)

settings = default_s()
settings['model-config.text_emb'] = ["char"]
settings['model-config.text_enc'] = ["bilstm"]
settings['model-config.text_project'] = [0]
settings['model-config.amr_emb'] = ["char"]
settings['model-config.amr_enc'] = ["lstm", "bilstm"]
settings['model-config.amr_project'] = [0]
settings['model-config.context'] = ['x', 'xy']
settings['model-config.prior'] = ['attn', 'unif']
settings['model-config.hidden_size'] = [100]
settings['batch-size'] = [4]
settings['lr'] = [1e-3]
settings['max-epoch'] = [20]
all_settings.append(settings)

settings = default_s()
settings['model-config.text_emb'] = ["char"]
settings['model-config.text_enc'] = ["bilstm"]
settings['model-config.text_project'] = [0]
settings['model-config.amr_emb'] = ["char"]
settings['model-config.amr_enc'] = ["tree_rnn"]
settings['model-config.amr_project'] = [0]
settings['model-config.context'] = ['x', 'xy']
settings['model-config.prior'] = ['attn', 'unif']
settings['model-config.hidden_size'] = [100]
settings['batch-size'] = [8]
settings['lr'] = [1e-3]
settings['max-epoch'] = [20]
all_settings.append(settings)

settings = default_s()
settings['model-config.text_emb'] = ["char"]
settings['model-config.text_enc'] = ["bilstm"]
settings['model-config.text_project'] = [0]
settings['model-config.amr_emb'] = ["char"]
settings['model-config.amr_enc'] = ["tree_rnn"]
settings['model-config.amr_project'] = [0]
settings['model-config.context'] = ['xy']
settings['model-config.context_2'] = ['x', 'e']
settings['model-config.lambda_context'] = [0.25, 0.5, 0.75]
settings['model-config.prior'] = ['attn']
settings['model-config.hidden_size'] = [100]
settings['batch-size'] = [8]
settings['lr'] = [1e-3]
settings['max-epoch'] = [20]
all_settings.append(settings)


def nested_set(o, k, v):
    parts = k.split('.', 1)

    if len(parts) == 1:
        o[k] = v
        return

    k, rest = parts
    if k not in o:
        o[k] = {}
    return nested_set(o[k], rest, v)


sofar = 0
for i_seed in range(seeds):
    i_exp = 0

    seen = set()

    for s in all_settings:
        for d in product_dict(s):
            # remove duplicates
            ukey = json.dumps(d, sort_keys=True)
            if ukey in seen:
                continue
            seen.add(ukey)

            # create seed-specific settings
            o = {}
            for k, v in d.items():
                nested_set(o, k, v)
            o['seed'] = np.random.randint(0, 100000000)
            o['name'] = 'version_{}_exp_{}_seed_{}'.format(version, i_exp, i_seed)
            print(json.dumps(o))
            i_exp += 1
            sofar += 1
        print('# {}'.format(sofar))

print('# {}'.format(sofar))

