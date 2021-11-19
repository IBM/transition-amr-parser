import argparse
import collections
import json
import os


errors = collections.Counter()


def main(args):
    with open(args.file_list) as f:
        file_list = f.read().strip().split('\n')
    print('file_list', len(file_list))

    def readfile(path):
        eval_path, train_path = path.split()


        path = eval_path
        slurm_out = os.path.join(path, 'slurm.out')
        eval_json = os.path.join(path, 'train.txt.no_wiki.eval.json')

        if not os.path.exists(slurm_out):
            raise ValueError

        flags_json = os.path.join(train_path, 'flags.json')
        with open(flags_json) as f:
            train_flags = json.loads(f.read())

        eval_flags = None
        try:
            flags_json = os.path.join(eval_path, 'flags.json')
            with open(flags_json) as f:
                eval_flags = json.loads(f.read())
        except:

            with open(slurm_out) as f:
                for i, line in enumerate(f):
                    if line.startswith('{'):
                        if line[1] == "'":
                            eval_flags = eval(line)
                        else:
                            eval_flags = json.loads(line.strip())
                        break

        train_slurm = os.path.join(train_path, 'slurm.out')

        if eval_flags is None:
            print('nothing found', slurm_out, train_slurm)
            raise ValueError

        if os.path.exists(slurm_out) and not os.path.exists(eval_json):
            print('possible error {} {} {} {}'.format(slurm_out, eval_flags['hostname'], train_slurm, train_flags['hostname']))
            errors['train-{}'.format(train_flags['hostname'])] += 1
            errors['eval-{}'.format(eval_flags['hostname'])] += 1

        if not os.path.exists(eval_json):
            raise ValueError

        # read eval_json
        with open(eval_json) as f:
            o = json.loads(f.read())
        o['path'] = path
        o['train_flags'] = train_flags
        o['eval_flags'] = eval_flags
        return o

    def try_map(items, func):
        for x in items:
            try:
                yield func(x)
            except ValueError:
                continue

    def groupby(data):
        groups = collections.defaultdict(list)

        for ex in data:
            groups[ex['train_flags']['log_dir']].append(ex)

        return groups

    for k, v in sorted(errors.items(), key=lambda x: x[1]):
        print(k, v)

    data = [x for x in try_map(file_list, readfile)]
    for ex in data:
        recall = ex['Corpus Recall using spans for gold']['recall']
        ex['recall'] = recall

    groups = groupby(data)

    for group in sorted(groups.values(), key=lambda x: max(map(lambda x: x['recall'], x))):
        print(group[0]['train_flags']['log_dir'])
        for ex in sorted(group, key=lambda x: x['path']):
            print(ex['recall'], ex['path'])

    print('data', len(data), 'groups', len(groups))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', default='eval_json.2021-11-05a.txt', type=str)
    args = parser.parse_args()

    main(args)


