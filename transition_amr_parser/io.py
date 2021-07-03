import re
import json
from collections import Counter
import subprocess
from transition_amr_parser.amr import JAMR_CorpusReader
import ast
import xml.etree.ElementTree as ET

import shutil
import numpy as np


def clbar(
    xy=None,  # list of (x, y) tuples or Counter
    x=None,
    y=None,
    ylim=(None, None),
    ncol=None,    # Max number of lines for display (defauly window size)
    # show only top and bottom values
    topx=None,
    botx=None,
    topy=None,
    boty=None,
    # normalize to sum to 1
    norm=False,
    xfilter=None,  # f(x) returns bool to not skip this example in display
    yform=None     # Function receiveing single y value returns string
):
    """Print data structure in command line"""
    # Sanity checks
    if x is None and y is None:
        if isinstance(xy, np.ndarray):
            labels = [f'{i}' for i in range(xy.shape[0])]
            xy = list(zip(labels, list(xy)))
        elif isinstance(xy, Counter):
            xy = [(str(x), y) for x, y in xy.items()]
        else:
            assert isinstance(xy, list), "Expected list of tuples"
            assert isinstance(xy[0], tuple), "Expected list of tuples"
    else:
        assert x is not None and y is not None
        assert isinstance(x, list)
        assert isinstance(y, list) or isinstance(y, np.ndarray)
        assert len(x) == len(list(y))
        xy = list(zip(x, y))

    # normalize
    if norm:
        z = sum([x[1] for x in xy])
        xy = [(k, v / z) for k, v in xy]
    # show only top x
    if topx is not None:
        xy = sorted(xy, key=lambda x: float(x[0]))[-topx:]
    if botx is not None:
        xy = sorted(xy, key=lambda x: float(x[0]))[:botx]
    if boty is not None:
        xy = sorted(xy, key=lambda x: x[1])[:boty]
    if topy is not None:
        xy = sorted(xy, key=lambda x: x[1])[-topy:]
    # print list of tuples
    # determine variables to fit data to command line
    x_data, y_data = zip(*xy)
    width = max([len(x) if x is not None else len('None') for x in x_data])
    number_width = max([len(f'{y}') for y in y_data])
    # max and min values
    if ylim[1] is not None:
        max_y_data = ylim[1]
    else:
        max_y_data = max(y_data)
    if ylim[0] is not None:
        min_y_data = ylim[0]
    else:
        min_y_data = min(y_data)
    # determine scaling factor from screen size
    data_range = max_y_data - min_y_data
    if ncol is None:
        ncol, _ = shutil.get_terminal_size((80, 20))
    max_size = ncol - width - number_width - 3
    scale = max_size / data_range
    # plot
    print()
    blank = ' '
    if yform:
        min_y_data_str = yform(min_y_data)
        print(f'{blank:<{width}}{min_y_data_str}')
    else:
        print(f'{blank:<{width}}{min_y_data}')
    for (x, y) in xy:

        # Filter example by x
        if xfilter and not xfilter(x):
            continue

        if y > max_y_data:
            # cropped bars
            num_col = int((ylim[1] - min_y_data) * scale)
            if num_col == 0:
                bar = ''
            else:
                half_width = (num_col // 2)
                if num_col % 2:
                    bar = '\u25A0' * (half_width - 1)
                    bar += '//'
                    bar += '\u25A0' * (half_width - 1)
                else:
                    bar = '\u25A0' * half_width
                    bar += '//'
                    bar += '\u25A0' * (half_width - 1)
        else:
            bar = '\u25A0' * int((y - min_y_data) * scale)
        if x is None:
            x = 'None'
        if yform:
            y = yform(y)
            print(f'{x:<{width}} {bar} {y}')
        else:
            print(f'{x:<{width}} {bar} {y}')
    print()


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


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


def read_action_scores(file_path):
    """
    Reads scores to judge the optimality of an action set, comprise

    sentence id (position in the original corpus)       1 int
    unormalized scores                                  3 int
    sequence normalized score e.g. smatch               1 float
    action sequence length                              1 int
    saved because of {score, length, None (original)}   1 str
    action sequence (tab separated)                     1 str (tab separated)

    TODO: Probability
    """
    action_scores = []
    with open(file_path) as fid:
        for line in fid:
            line = line.strip()
            items = list(map(int, line.split()[:4]))
            items.append(float(line.split()[4]))
            items.append(int(line.split()[5]))
            items.append(
                None if line.split()[6] == 'None' else line.split()[6]
            )
            if line.split()[7][0] == '[':
                # backwards compatibility fix
                items.append(ast.literal_eval(" ".join(line.split()[7:])))
            else:
                items.append(line.split()[7:])
            action_scores.append(items)

    return action_scores


def write_action_scores(file_path, action_scores):
    """
    Writes scores to judge the optimality of an action set, comprise

    sentence id (position in the original corpus)       1 int
    unormalized scores                                  3 int
    sequence normalized score e.g. smatch               1 float
    action sequence length                              1 int
    saved because of {score, length, None (original)}   1 str
    action sequence (tab separated)                     1 str (tab separated)

    TODO: Probability
    """

    with open(file_path, 'w') as fid:
        for items in action_scores:
            sid = items[0]
            score = items[1:4]
            smatch = items[4]
            length = items[5]
            reason = items[6]
            actions = items[7]
            if actions is not None:
                actions = '\t'.join(actions)
            fid.write(
                f'{sid} {score[0]} {score[1]} {score[2]} {smatch} {length}'
                f' {reason} {actions}\n'
            )


def read_amr(in_amr, unicode_fixes=False):

    corpus = JAMR_CorpusReader()
    corpus.load_amrs(in_amr)

    if unicode_fixes:

        # Replacement rules for unicode chartacters
        replacement_rules = {
            'ˈtʃærɪti': 'charity',
            '\x96': '_',
            '⊙': 'O'
        }

        # FIXME: normalization shold be more robust. Right now use the tokens
        # of the amr inside the oracle. This is why we need to normalize them.
        for idx, amr in enumerate(corpus.amrs):
            new_tokens = []
            for token in amr.tokens:
                forbidden = [x for x in replacement_rules.keys() if x in token]
                if forbidden:
                    token = token.replace(
                        forbidden[0],
                        replacement_rules[forbidden[0]]
                     )
                new_tokens.append(token)
            amr.tokens = new_tokens

    return corpus


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
