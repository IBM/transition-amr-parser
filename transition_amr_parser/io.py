import re
import json
from collections import Counter
from transition_amr_parser.amr import JAMR_CorpusReader
import ast


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
            reason = items [6]
            actions = items[7]
            if actions is not None:
                actions = '\t'.join(actions)
            fid.write(
                f'{sid} {score[0]} {score[1]} {score[2]} {smatch} {length} {reason} {actions}\n'
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
