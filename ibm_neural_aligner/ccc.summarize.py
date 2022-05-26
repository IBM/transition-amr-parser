import argparse
import json
import os
import collections
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/dccstor/ykt-parse/SHARED/misc/adrozdov/log/align', type=str)
parser.add_argument('--output', default='summary-output', type=str)
parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--archive', action='store_true')
parser.add_argument('--read-train', action='store_true')
args = parser.parse_args()


class NoDataError(Exception):
    pass


class Exp:
    @staticmethod
    def from_logdir(logdir, read_train=False):
        exp = Exp(logdir)
        exp.read_flags()
        exp.read_metrics()
        if read_train:
            exp.read_train()
        return exp

    def __init__(self, logdir):
        self.logdir = logdir
        self.amr2 = None

    def trim(self):
        epoch = self.recall['argmax']
        path = os.path.join(self.logdir, 'model.epoch_{}.pt'.format(epoch))
        new_path = os.path.join(self.logdir, 'model.best.recall.pt')
        if not os.path.exists(new_path):
            os.system('cp {} {}'.format(path, new_path))

        epoch = self.val_loss['argmin']
        path = os.path.join(self.logdir, 'model.epoch_{}.pt'.format(epoch))
        new_path = os.path.join(self.logdir, 'model.best.val_loss.pt')
        if not os.path.exists(new_path):
            os.system('cp {} {}'.format(path, new_path))

        for epoch in range(len(self.recall['values'])):
            path = os.path.join(self.logdir, 'model.epoch_{}.pt'.format(epoch))
            if os.path.exists(path):
                os.system('rm {}'.format(path))

    @property
    def key(self):
        cfg = self.flags.copy()
        cfg['log_dir'] = None
        cfg['name'] = None
        cfg['seed'] = None
        key = json.dumps(cfg, sort_keys=True)
        return key

    def read_flags(self):
        path = os.path.join(self.logdir, 'flags.json')
        if not os.path.exists(path):
            raise NoDataError

        with open(path) as f:
            self.flags = json.loads(f.read())

        self.flags['model_config'] = json.loads(self.flags['model_config'])

        def default_set(o, k, v):
            if k not in o:
                o[k] = v

        default_set(self.flags['model_config'], 'text_emb', None)
        default_set(self.flags['model_config'], 'amr_emb', None)
        default_set(self.flags['model_config'], 'hidden_size', None)

        assert isinstance(self.flags, dict)
        assert isinstance(self.flags['model_config'], dict)

    def read_train(self):
        path = os.path.join(self.logdir + '_write_amr2', 'alignment.trn.out.pred.eval')
        with open(path) as f:
            data = json.loads(f.read())
        self.amr2 = data

    def read_metrics(self):
        data_list = collections.defaultdict(list)
        epoch = 0
        while True:
            path = os.path.join(self.logdir, 'model.epoch_{}.metrics'.format(epoch))
            if not os.path.exists(path):
                break
            with open(path) as f:
                data = json.loads(f.read())
            for k, v in data.items():
                data_list[k].append(v)
            epoch += 1

        if len(data_list) == 0:
            raise NoDataError

        self.data_list = data_list

        m = {}

        for k, v in data_list.items():
            if len(v) == 0:
                raise NoDataError
            v = np.array(v)
            argmax = np.argmax(v).item()
            argmin = np.argmin(v).item()
            valmax = v[argmax].item()
            valmin = v[argmin].item()

            o = {}
            o['argmax'] = argmax
            o['argmin'] = argmin
            o['valmax'] = valmax
            o['valmin'] = valmin
            o['values'] = v

            m[k] = o

        self.m = m


default_groupby = (
    'model_config.text_emb',
    'model_config.text_enc',
    'model_config.amr_emb',
    'model_config.amr_enc',
    'model_config.hidden_size',
    'batch_size',
    'lr'
)

def nested_get(o, k):
    assert isinstance(o, dict), (o, type(o), k)
    parts = k.split('.', 1)
    if len(parts) == 1:
        return o[k]
    k, rest = parts
    return nested_get(o[k], rest)


class Summary:
    def __init__(self):
        self.c = collections.defaultdict(list)
        self.flags = {}

    def add(self, exp):
        if exp.key not in self.flags:
            self.flags[exp.key] = exp.flags
        self.c[exp.key].append(exp)

    def get_groupkey(self, groupby, flags):
        return tuple([nested_get(flags, x) for x in groupby])

    def summarize(self, groupby=default_groupby):
        g = collections.defaultdict(list)
        for k, exp_list in self.c.items():
            flags = self.flags[k]
            groupkey = self.get_groupkey(groupby, flags)
            for exp in exp_list:
                g[groupkey].append(exp)

        for k, exp_list in g.items():

            info = ' '.join([str(x) for x in k])
            recall = np.array([e.recall['valmax'] for e in exp_list])
            epoch = np.array([e.recall['argmax'] for e in exp_list])
            seeds = len(exp_list)

            print(f'{info} {seeds} {recall.max():.3f} {epoch[np.argmax(recall)]} {recall.mean():.3f} {epoch.mean():.1f}')

    def summarize_amr2(self):

        exp_list = []
        for e in self.c.values():
            for ee in e:
                exp_list.append(ee)

        print(len(exp_list))

        values = {}
        values['amr2_trn_recall'] = [e.amr2['Corpus Recall using spans for gold']['recall'] for e in exp_list]
        values['amr2_val_recall_seen'] = [e.m['val_0_recall']['valmax'] for e in exp_list]
        values['amr2_val_recall_seen_epoch'] = [e.m['val_0_recall']['argmax'] for e in exp_list]
        values['amr2_val_loss_seen'] = [e.m['val_0_loss']['values'][epoch] for epoch, e in zip(values['amr2_val_recall_seen_epoch'], exp_list)]
        index = collections.OrderedDict()
        k = 'amr2_trn_recall'
        index[k] = np.argsort(values[k])[::-1]
        k = 'amr2_val_recall_seen'
        index[k] = np.argsort(values[k])[::-1]
        k = 'amr2_val_loss_seen'
        index[k] = np.argsort(values[k])

        for k in list(index.keys()):
            index[k] = index[k].tolist()

        sort_k = 'amr2_trn_recall'
        for i, ix in enumerate(index[sort_k]):
            e = exp_list[ix]

            print(i)
            print(e.flags['log_dir'])
            print(e.flags)

            epoch = values['amr2_val_recall_seen_epoch'][ix]
            print('epoch = {}'.format(epoch))

            for k in index.keys():
                i_ = index[k].index(ix)
                v = values[k][ix]
                assert index[k][i_] == ix

                if sort_k == k:
                    assert i == i_

                print('{} = {:.3f}, rank = {}'.format(k, v, i_))
            print('')

    def summarize_verbose(self):

        keys = ['val_0_ppl', 'val_0_recall']
        keys2 = ['min', 'max', 'min', 'max']

        sortby = ('val_0_recall', 'valmax', -1)

        exp_list = []
        for e in self.c.values():
            for ee in e:
                exp_list.append(ee)

        sortkeys = [e.m[sortby[0]][sortby[1]] * sortby[2] for e in exp_list]
        index = np.argsort(sortkeys)

        for i in index:
            e = exp_list[i]

            print(e.flags['log_dir'])
            print(e.flags)
            
            for key, key2 in zip(keys, keys2):
                o = e.m[key]
                val = o['val{}'.format(key2)]
                arg = o['arg{}'.format(key2)]
                print(key, arg, round(val, 3))
            print('')

    def archive(self, output):
        os.system('mkdir -p {}'.format(output))

        exp_list = []
        for e in self.c.values():
            for ee in e:
                exp_list.append(ee)

        for e in exp_list:

            logdir = e.flags['log_dir']

            prefix = e.flags['name']

            paths = []
            paths.append([os.path.join(logdir, 'flags.json'), os.path.join(output, prefix + '.' + 'flags.json')])

            epoch = e.recall['argmax']
            paths.append([os.path.join(logdir, 'alignment.epoch_{}.val.out.pred'.format(epoch)), os.path.join(output, prefix + '.' + 'alignment.best_recall.pred')])
            paths.append([os.path.join(logdir, 'alignment.epoch_{}.val.out.gold'.format(epoch)), os.path.join(output, prefix + '.' + 'alignment.best_recall.gold')])

            epoch = e.val_loss['argmin']
            paths.append([os.path.join(logdir, 'alignment.epoch_{}.val.out.pred'.format(epoch)), os.path.join(output, prefix + '.' + 'alignment.best_val_loss.pred')])
            paths.append([os.path.join(logdir, 'alignment.epoch_{}.val.out.gold'.format(epoch)), os.path.join(output, prefix + '.' + 'alignment.best_val_loss.gold')])

            for cp_from, cp_to in paths:
                os.system('cp {} {}'.format(cp_from, cp_to))




def main():
    stats = collections.Counter()
    s = Summary()

    prefix = 'exp' if not args.prefix else args.prefix

    for fn in os.listdir(args.path):
        if not fn.startswith(prefix):
            continue
        if 'write' in fn:
            continue
        logdir = os.path.join(args.path, fn)
        try:
            exp = Exp.from_logdir(logdir, read_train=args.read_train)
            stats['found'] += 1
        except FileNotFoundError:
            stats['skipped-file_not_found'] += 1
            continue
        except NoDataError:
            stats['skipped'] += 1
            continue
        s.add(exp)

    if args.read_train:
        s.summarize_amr2()
    else:
        s.summarize_verbose()

    if args.archive:
        s.archive(args.output)

    print(stats)

main()
