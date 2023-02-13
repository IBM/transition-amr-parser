import re
import os
from glob import glob
import json
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from collections import Counter
from transition_amr_parser.amr import AMR
from dateutil.parser import parse


def get_score_from_log(file_path, score_name):

    smatch_results_re = re.compile(r'^F-score: ([0-9\.]+)')

    results = [None]

    if 'smatch' in score_name:
        regex = smatch_results_re
    else:
        raise Exception(f'Unknown score type {score_name}')

    with open(file_path) as fid:
        for line in fid:
            if regex.match(line):
                results = regex.match(line).groups()
                results = [100*float(x) for x in results]
                break

    return results


def read_train_log(seed_folder, config_name):

    train_info_regex = re.compile(
        r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) '
        r'\| INFO \| train \| (.*)'
    )
    valid_info_regex = re.compile(
        r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) '
        r'\| INFO \| valid \| (.*)'
    )

    seed_data = []
    for log_file in glob(f'{seed_folder}/tr-*.stdout'):
        with open(log_file) as fid:
            for line in fid:
                if train_info_regex.match(line):
                    date_str, json_info = \
                        train_info_regex.match(line).groups()
                    item = json.loads(json_info)
                    item['timestamp'] = parse(date_str)
                    item['experiment_key'] = config_name
                    item['set'] = 'train'
                    item['name'] = config_name
                    seed_data.append(item)

                elif valid_info_regex.match(line):
                    date_str, json_info = \
                        valid_info_regex.match(line).groups()
                    item = json.loads(json_info)
                    item['timestamp'] = parse(date_str)
                    item['experiment_key'] = config_name
                    item['set'] = 'valid'
                    item['name'] = config_name
                    seed_data.append(item)

    # TODO: compute time between epochs
    train_seed_data = [x for x in seed_data if 'train_nll_loss' in x]
    train_data_sort = sorted(train_seed_data, key=lambda x: x['epoch'])
    if train_data_sort == []:
        return []
    train_data_sort[0]['epoch_time'] = None
    prev = [train_data_sort[0]['epoch'], train_data_sort[0]['timestamp']]
    new_data = [train_data_sort[0]]
    for item in train_data_sort[1:]:
        time_delta = item['timestamp'] - prev[1]
        epoch_delta = item['epoch'] - prev[0]
        if epoch_delta:
            item['epoch_time'] = time_delta.total_seconds() / epoch_delta
            prev = [item['epoch'], item['timestamp']]
        else:
            item['epoch_time'] = 0
        new_data.append(item)

    return new_data


def read_experiment(config):

    config_name = os.path.basename(config)
    config_env_vars = read_config_variables(config)
    model_folder = config_env_vars['MODEL_FOLDER']
    seeds = config_env_vars['SEEDS'].split()

    exp_data = []
    for seed in seeds:
        seed_folder = f'{model_folder}seed{seed}'

        # read info from logs
        exp_data.extend(read_train_log(seed_folder, config_name))

        # read validation decoding scores
        eval_metric = config_env_vars['EVAL_METRIC']
        validation_folder = f'{seed_folder}/epoch_tests/'
        for epoch in range(int(config_env_vars['MAX_EPOCH'])):
            results_file = \
                f'{validation_folder}/dec-checkpoint{epoch}.{eval_metric}'
            if os.path.isfile(results_file):
                score = get_score_from_log(results_file, eval_metric)[0]
                exp_data.append({
                    'epoch': epoch,
                    'epoch_time': None,
                    'set': 'valid-dec',
                    'score': score,
                    'experiment_key': config_name,
                    'name': config_name
                })

    return exp_data


def write_neural_alignments(out_alignment_probs, aligned_amrs, joints):
    assert len(aligned_amrs) == len(joints)
    with open(out_alignment_probs, 'w') as fid:
        for index in range(len(joints)):
            # index, nodes, node ids, tokens
            fid.write(f'{index}\n')
            nodes = ' '.join(aligned_amrs[index].nodes.values())
            fid.write(f'{nodes}\n')
            ids = ' '.join(aligned_amrs[index].nodes.keys())
            fid.write(f'{ids}\n')
            tokens = ' '.join(aligned_amrs[index].tokens)
            fid.write(f'{tokens}\n')
            # probabilities
            num_tokens, num_nodes = joints[index].shape
            for j in range(num_nodes):
                for i in range(num_tokens):
                    fid.write(f'{j} {i} {joints[index][i, j]:e}\n')
            fid.write(f'\n')


def read_neural_alignments_from_memmap(path, corpus):
    # TODO: Verify hash, which ensures dataset and node order is the same.
    sizes = list(map(lambda x: len(x.tokens) * len(x.nodes), corpus))
    assert all([size > 0 for size in sizes])
    total_size = sum(sizes)
    offsets = np.zeros(len(corpus), dtype=np.int)
    offsets[1:] = np.cumsum(sizes[:-1])

    np_align_dist = np.memmap(path, dtype=np.float32,
                              shape=(total_size, 1), mode='r')
    align_dist = np.zeros((total_size, 1), dtype=np.float32)
    align_dist[:] = np_align_dist[:]

    alignments = []
    for idx, amr in enumerate(corpus):
        offset = offsets[idx]
        size = sizes[idx]
        assert size == len(amr.tokens) * len(amr.nodes)

        p_node_and_token = align_dist[offset:offset +
                                      size].reshape(len(amr.nodes), len(amr.tokens))
        example_id = idx
        node_short_id = None
        text_tokens = None

        alignments.append(dict(
            example_id=example_id,
            node_short_id=node_short_id,
            text_tokens=text_tokens,
            p_node_and_token=p_node_and_token
        ))

    return alignments


def read_neural_alignments(alignments_file):

    alignments = []
    sentence_alignment = []
    with open(alignments_file, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line == '':

                # example_id
                # node_names
                # node_short_id
                # text_tokens
                # i j posterior[i, j]

                example_id, node_names, node_short_id, text_tokens = \
                    sentence_alignment[:4]

                # FIXME: Some node name have "some space" need to be saped
                # num_nodes = len(node_names.split())
                num_nodes = len(node_short_id.split())
                num_tokens = len(text_tokens.split())
                p_node_and_token = np.zeros((num_nodes, num_tokens))
                node_indices = set()
                token_indices = set()
                for pair in sentence_alignment[4:]:
                    node_idx, token_idx, prob = pair.split()
                    node_idx, token_idx = int(node_idx), int(token_idx)
                    p_node_and_token[node_idx, token_idx] = float(prob)
                    node_indices.add(node_idx)
                    token_indices.add(token_idx)

                # sanity check
                assert list(node_indices) == list(range(num_nodes))
                assert list(token_indices) == list(range(num_tokens))
                # assert len(node_short_id.split()) == len(node_names.split())

                alignments.append(dict(
                    example_id=example_id,
                    node_short_id=node_short_id.split(),
                    # node_names=node_names.split(),
                    text_tokens=text_tokens.split(),
                    p_node_and_token=p_node_and_token
                ))
                sentence_alignment = []
            else:
                sentence_alignment.append(line)

    return alignments


def read_amr(file_path, jamr=False, generate=False):
    if generate:
        # yields each AMR, faster but non sequential
        return amr_generator(file_path, jamr=jamr)
    else:
        return amr_iterator(file_path, jamr=jamr)


def amr_iterator(file_path, jamr=False):
    '''
    Read AMRs in PENMAN+ISI-alignments or JAMR+alignments (ibm_format=True)
    '''

    amrs = []
    # loop over blocks separated by whitespace
    tqdm_iterator = read_blocks(file_path)
    num_amr = len(tqdm_iterator)
    for index, raw_amr in enumerate(tqdm_iterator):

        if jamr:
            # From JAMR plus IBMs alignment format (DEPRECATED)
            amrs.append(AMR.from_metadata(raw_amr))
        else:
            # from penman
            amrs.append(AMR.from_penman(raw_amr))

        tqdm_iterator.set_description(f'Reading AMRs {index+1}/{num_amr}')

    return amrs


def amr_generator(file_path, jamr=False):
    '''
    Read AMRs in PENMAN+ISI-alignments or JAMR+alignments (ibm_format=True)

    (tokenize is deprecated)
    '''

    # loop over blocks separated by whitespace
    tqdm_iterator = read_blocks(file_path)
    num_amr = len(tqdm_iterator)
    for index, raw_amr in enumerate(tqdm_iterator):

        if jamr:
            # From JAMR plus IBMs alignment format (DEPRECATED)
            yield AMR.from_metadata(raw_amr)
        else:
            # From penman
            yield AMR.from_penman(raw_amr)

        tqdm_iterator.set_description(f'Reading AMRs {index+1}/{num_amr}')


def generate_blocks(file_path, bar=True, desc=None):
    '''
    Reads text file, returns chunks separated by empty line
    '''

    # to measure progress with a generator get the size first
    if bar:
        with open(file_path) as fid:
            num_blocks = len([x for x in fid])

    # display a progress bar
    def pbar(x):
        if bar:
            return tqdm(x, desc=desc, total=num_blocks)
        else:
            return x

    # read blocks
    with open(file_path) as fid:
        block = ''
        for line in pbar(fid):
            if line.strip() == '':
                yield block
                block = ''
            else:
                block += line


def read_blocks(file_path, return_tqdm=True):
    '''
    Reads text file, returns chunks separated by empty line
    '''

    # read blocks
    with open(file_path) as fid:
        block = ''
        blocks = []
        for line in fid:
            if line.strip() == '':
                blocks.append(block)
                block = ''
            else:
                block += line

    if return_tqdm:
        return tqdm(blocks, leave=False)
    else:
        return blocks


def read_penman_metadata(penman_text):
    '''
    Read metadata from penman into dictionary
    '''
    metadata = {}
    for line in penman_text.split('\n'):
        if not line.lstrip().startswith('#'):
            continue
        for field in line.split('::'):
            if field.strip() != '#':
                items = field.strip().split()
                if len(items) == 2:
                    metadata[items[0]] = items[1]
                elif len(items) == 1:
                    metadata[items[0]] = True
                else:
                    raise Exception(f'{penman_text}')
    return metadata


def read_penmans(amr_files):
    '''
    Returns generator of multiple versions of same amr in penman notation
    '''

    # get number of AMRs and check all match
    sizes = []
    for amr_file in amr_files:
        with open(amr_file) as fid:
            num_amrs = 0
            for line in fid:
                if line.strip() == '':
                    num_amrs += 1
        sizes.append(num_amrs)

    if len(set(sizes)) != 1:
        print(sizes)
        raise Exception('amr files must have same number of AMRs')

    def amr_generator():
        fids = [open(dfile) for dfile in amr_files]
        # loop over amrs in the corpus
        for n in range(sizes[0]):
            # loop over versions of the same AMR
            amrs = []
            for fid in fids:
                penman = ''
                # consume lines in this file until end of a single AMR
                for line in fid:
                    if line.strip() == '':
                        # completed this AMR
                        amrs.append(penman)
                        break
                    else:
                        # accumulate
                        penman += line

            # return all versions of this AMR
            yield amrs

    # return as tqdm generator
    return tqdm(amr_generator(), total=sizes[0])


def read_frame(xml_file):
    '''
    Read probpank XML
    '''

    root = ET.parse(xml_file).getroot()
    propbank = {}
    for predicate in root.findall('predicate'):
        lemma = predicate.attrib['lemma']
        for roleset_data in predicate.findall('roleset'):

            # ID of the role e.g. run.01
            pred_id = roleset_data.attrib['id']

            # basic meta-data
            propbank[pred_id] = {
                'lemma': lemma,
                'description': roleset_data.attrib['name']
            }

            # alias
            propbank[pred_id]['aliases'] = []
            for aliases in roleset_data.findall('aliases'):
                for alias in aliases:
                    propbank[pred_id]['aliases'].append(alias.text)

            # roles
            propbank[pred_id]['roles'] = {}
            for roles in roleset_data.findall('roles'):
                for role in roles:
                    if role.tag == 'note':
                        continue
                    number = role.attrib['n']
                    propbank[pred_id]['roles'][f'ARG{number}'] = role.attrib

            # examples
            propbank[pred_id]['examples'] = []
            for examples in roleset_data.findall('example'):
                sentence = examples.findall('text')
                assert len(sentence) == 1
                sentence = sentence[0].text
                tokens = [x.text for x in examples.findall('rel')]
                args = []
                for x in examples.findall('arg'):
                    args.append(x.attrib)
                    args[-1].update({'text': x.text})
                propbank[pred_id]['examples'].append({
                    'sentence': sentence,
                    'tokens': tokens,
                    'args': args
                })

    return propbank


def read_config_variables(config_path):
    """
    Read an experiment bash config (e.g. the ones in configs/ )
    """

    # Read variables into dict
    # Read all variables of this pattern
    variable_regex = re.compile('^ *([A-Za-z0-9_]+)=.*$')
    # find variables in text and prepare evaluation script
    bash_script = f'source {config_path};'
    with open(config_path) as fid:
        for line in fid:
            if variable_regex.match(line.strip()):
                varname = variable_regex.match(line.strip()).groups()[0]
                bash_script += f'echo "{varname}=${varname}";'
    # Execute script to get variable's value
    config_env_vars = {}
    proc = subprocess.Popen(
        bash_script, stdout=subprocess.PIPE, shell=True, executable='/bin/bash'
    )
    for line in proc.stdout:
        (key, _, value) = line.decode('utf-8').strip().partition("=")
        config_env_vars[key] = value

    return config_env_vars


def read_rule_stats(rule_stats_json):
    with open(rule_stats_json) as fid:
        rule_stats = json.loads(fid.read())
    # convert to counters
    rule_stats['possible_predicates'] = \
        Counter(rule_stats['possible_predicates'])
    rule_stats['action_vocabulary'] = Counter(rule_stats['action_vocabulary'])
    return rule_stats


def write_rule_stats(rule_stats_json, content):
    with open(rule_stats_json, 'w') as fid:
        fid.write(json.dumps(content))


def read_propbank(propbank_file):

    # Read frame argument description
    arguments_by_sense = {}
    with open(propbank_file) as fid:
        for line in fid:
            line = line.rstrip()
            sense = line.split()[0]
            arguments = [
                re.match('^(ARG.+):$', x).groups()[0]
                for x in line.split()[1:] if re.match('^(ARG.+):$', x)
            ]
            arguments_by_sense[sense] = arguments

    return arguments_by_sense


def writer(file_path, add_return=False):
    """
    Returns a writer that writes to file_path if it is not None, does nothing
    otherwise

    calling the writed without arguments will close the file
    """
    if file_path:
        # Erase file
        fid = open(file_path, 'w+')
        fid.close()
        # open for appending
        fid = open(file_path, 'a+', encoding='utf8')
    else:
        fid = None

    def append_data(content=None):
        """writes to open file"""
        if fid:
            if content is None:
                fid.close()
            else:
                if add_return:
                    fid.write(content + '\n')
                else:
                    fid.write(content)

    return append_data


def tokenized_sentences_egenerator(file_path):
    with open(file_path) as fid:
        for line in fid:
            yield line.rstrip().split()


def read_tokenized_sentences(file_path, separator=' '):
    sentences = []
    with open(file_path) as fid:
        for line in fid:
            sentences.append(line.rstrip().split(separator))
    return sentences


def write_tokenized_sentences(file_path, content, separator=' '):
    with open(file_path, 'w') as fid:
        for line in content:
            line = [str(x) for x in line]
            fid.write(f'{separator.join(line)}\n')


def read_sentences(file_path, add_root_token=False):
    sentences = []
    with open(file_path) as fid:
        for line in fid:
            line = line.rstrip()
            if add_root_token:
                line = line + " <ROOT>"
            sentences.append(line)
    return sentences
