import argparse
import collections
import json
import os


errors = collections.Counter()

class DidNotEval(ValueError):
    pass

class DidNotTrain(ValueError):
    pass

class NoEvalMetrics(ValueError):
    pass

class NoTrainMetrics(ValueError):
    pass

class NoModel(ValueError):
    pass


def read_train_flags(log_dir):
    path = os.path.join(log_dir, 'flags.json')
    if not os.path.exists(path):
        raise DidNotTrain
    with open(path) as f:
        return json.loads(f.read())

def read_metrics(log_dir, flags):
    max_epoch = flags['max_epoch']

    metrics = []
    for epoch in range(max_epoch):
        path = os.path.join(log_dir, 'model.epoch_{}.metrics'.format(epoch))
        if not os.path.exists(path):
            break
        with open(path) as f:
            try:
                m = json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print('json error', path)
                break
            m['epoch'] = epoch
            metrics.append(m)
    return metrics


def maybe_func(func, exceptions, default_val=None):
    try:
        return func()
    except exceptions as e:
        return default_val

def read_one(log_dir):
    exp = {}
    exp['flags'] = read_train_flags(log_dir)
    exp['metrics'] = read_metrics(log_dir, exp['flags'])

    # best ppl
    if len(exp['metrics']) == 0:
        raise NoTrainMetrics

    best_m = min(exp['metrics'], key=lambda x: x['val_0_ppl'])
    exp['best_val_ppl'] = best_m['val_0_ppl']
    exp['best_val_ppl_epoch'] = best_m['epoch']


    return exp


def read_all_eval(log_dir, eval_file_list, train):
    if train is not None:
        train_epochs = set([m['epoch'] for m in train['metrics']])
    else:
        train_epochs = set([])

    exp = {}
    exp['metrics'] = []
    exp['metrics_not_found'] = []
    for eval_dir in eval_file_list:
        if 'latest' in eval_dir:
            continue
        path = os.path.join(eval_dir, 'train.aligned.txt.eval.json')
        epoch = int(path.split('epoch_')[1].split('/')[0])
        if not os.path.exists(path):
            if epoch in train_epochs:
                exp['metrics_not_found'].append(epoch)
            continue
        with open(path) as f:
            res = json.loads(f.read())
        res['path'] = path
        res['epoch'] = epoch
        res['recall'] = res['Corpus Recall using spans for gold']['recall']
        exp['metrics'].append(res)

    if len(exp['metrics']) == 0:
        raise NoEvalMetrics

    # best recall
    best_m = max(exp['metrics'], key=lambda x: x['recall'])
    exp['best_recall'] = best_m['recall']
    exp['best_recall_epoch'] = best_m['epoch']

    return exp

def read_all(file_list):
    data = {}
    for eval_dir, log_dir in file_list:
        if log_dir not in data:
            data[log_dir] = {}
            data[log_dir]['eval'] = []
        data[log_dir]['eval'].append(eval_dir)

    for log_dir in sorted(data.keys()):
        data[log_dir]['train'] = maybe_func(lambda: read_one(log_dir), (DidNotTrain, NoTrainMetrics, ), None)
        yield data, log_dir, 'train'

    for log_dir in sorted(data.keys()):
        data[log_dir]['eval_results'] = maybe_func(lambda: read_all_eval(log_dir, data[log_dir]['eval'], data[log_dir]['train']), (NoEvalMetrics, ), None)
        yield data, log_dir, 'eval_results'


def main(args):
    with open(args.file_list) as f:
        file_list = f.read().strip().split('\n')
        file_list = [x.split() for x in file_list]
    print('file_list', len(file_list))

    for data, k, msg in read_all(file_list):
        exp = data[k]

        if msg == 'train':

            # train
            train = exp['train']
            if train is None:
                print('train', k, None)
            else:
                print('train', k, train['best_val_ppl_epoch'], train['best_val_ppl'])

        elif msg == 'eval_results':

            eval_results = exp['eval_results']
            if eval_results is None:
                print('eval', k, None)
            else:
                print('eval', k, eval_results['best_recall_epoch'], eval_results['best_recall'])
            train = exp['train']

    for k in sorted(data.keys()):
        exp = data[k]

        if True:
            train = exp['train']
            eval_results = exp['eval_results']

            if train is not None and eval_results is not None:
                if eval_results['best_recall'] < 0.8:
                    continue
                print('both')
                print('train', k, train['best_val_ppl_epoch'], train['best_val_ppl'], len(train['metrics']))
                print('eval', k, eval_results['best_recall_epoch'], eval_results['best_recall'], len(eval_results['metrics']))

                for m in eval_results['metrics']:
                    if m['epoch'] >= train['best_val_ppl_epoch']:
                        not_found = [x for x in eval_results['metrics_not_found'] if x < m['epoch']]
                        print('early', k, m['epoch'], m['recall'], not_found)
                        break

    k = min(data.keys(), key=lambda x: data[x]['train']['best_val_ppl'] if data[x]['train'] is not None else 100000)
    train = data[k]['train']
    eval_results = data[k]['eval_results']
    print('best')
    if train is not None:
        print('train', k, train['best_val_ppl_epoch'], train['best_val_ppl'], len(train['metrics']))
    if eval_results is not None:
        print('eval', k, eval_results['best_recall_epoch'], eval_results['best_recall'], len(eval_results['metrics']))

    k = max(data.keys(), key=lambda x: data[x]['eval_results']['best_recall'] if data[x]['eval_results'] is not None else 0)
    train = data[k]['train']
    eval_results = data[k]['eval_results']
    print('best')
    if train is not None:
        print('train', k, train['best_val_ppl_epoch'], train['best_val_ppl'], len(train['metrics']))
    if eval_results is not None:
        print('eval', k, eval_results['best_recall_epoch'], eval_results['best_recall'], len(eval_results['metrics']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', default='eval_json.2021-11-05a.txt', type=str)
    args = parser.parse_args()

    print(args.__dict__)

    main(args)


