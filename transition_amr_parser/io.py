import re
import json
from collections import Counter


def read_rule_stats(rule_stats_json):
    with open(rule_stats_json) as fid:
        rule_stats = json.loads(fid.read())
    # convert to counters
    rule_stats['possible_predicates'] = Counter(rule_stats['possible_predicates'])
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
