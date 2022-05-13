import re
import json
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from collections import Counter
from transition_amr_parser.amr import AMR


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

    np_align_dist = np.memmap(path, dtype=np.float32, shape=(total_size, 1), mode='r')
    align_dist = np.zeros((total_size, 1), dtype=np.float32)
    align_dist[:] = np_align_dist[:]

    alignments = []
    for idx, amr in enumerate(corpus):
        offset = offsets[idx]
        size = sizes[idx]
        assert size == len(amr.tokens) * len(amr.nodes)

        p_node_and_token = align_dist[offset:offset+size].reshape(len(amr.nodes), len(amr.tokens))
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


def protected_tokenizer(sentence_string, simple=False):

    if simple:
        # simplest possible tokenizer
        # split by these symbols
        sep_re = re.compile(r'[\.,;:?!"\' \(\)\[\]\{\}]')
        return simple_tokenizer(sentence_string, sep_re)
    else:
        # imitates JAMR (97% sentece acc on AMR2.0)
        # split by these symbols
        # TODO: Do we really need to split by - ?
        sep_re = re.compile(r'[/~\*%\.,;:?!"\' \(\)\[\]\{\}-]')
        return jamr_like_tokenizer(sentence_string, sep_re)


def jamr_like_tokenizer(sentence_string, sep_re):

    # quote normalization
    sentence_string = sentence_string.replace('``', '"')
    sentence_string = sentence_string.replace("''", '"')
    sentence_string = sentence_string.replace("“", '"')

    # currency normalization
    #sentence_string = sentence_string.replace("£", 'GBP')

    # Do not split these strings
    protected_re = re.compile("|".join([
        # URLs (this conflicts with many other cases, we should normalize URLs
        # a priri both on text and AMR)
        #r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
        #
        r'[0-9][0-9,\.:/-]+[0-9]',         # quantities, time, dates
        r'^[0-9][\.](?!\w)',               # enumerate
        r'\b[A-Za-z][\.](?!\w)',           # itemize
        r'\b([A-Z]\.)+[A-Z]?',             # acronym with periods (e.g. U.S.)
        r'!+|\?+|-+|\.+',                  # emphatic
        r'etc\.|i\.e\.|e\.g\.|v\.s\.|p\.s\.|ex\.',     # latin abbreviations
        r'\b[Nn]o\.|\bUS\$|\b[Mm]r\.',     # ...
        r'\b[Mm]s\.|\bSt\.|\bsr\.|a\.m\.', # other abbreviations
        r':\)|:\(',                        # basic emoticons
        # contractions
        r'[A-Za-z]+\'[A-Za-z]{3,}',        # quotes inside words
        r'n\'t(?!\w)',                     # negative contraction (needed?)
        r'\'m(?!\w)',                      # other contractions
        r'\'ve(?!\w)',                     # other contractions
        r'\'ll(?!\w)',                     # other contractions
        r'\'d(?!\w)',                      # other contractions
        #r'\'t(?!\w)'                      # other contractions
        r'\'re(?!\w)',                     # other contractions
        r'\'s(?!\w)',                      # saxon genitive
        #
        r'<<|>>',                          # weird symbols
        #
        r'Al-[a-zA-z]+|al-[a-zA-z]+',      # Arabic article
        # months
        r'Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.'
    ]))

    # iterate over protected sequences, tokenize unprotected and append
    # protected strings
    tokens = []
    positions = []
    start = 0
    for point in protected_re.finditer(sentence_string):

        # extract preceeding and protected strings
        end = point.start()
        preceeding_str = sentence_string[start:end]
        protected_str = sentence_string[end:point.end()]

        if preceeding_str:
            # tokenize preceeding string keep protected string as is
            for token, (start2, end2) in zip(
                *simple_tokenizer(preceeding_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))
        tokens.append(protected_str)
        positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        ending_str = sentence_string[start:end]
        if ending_str.strip():
            for token, (start2, end2) in zip(
                *simple_tokenizer(ending_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))

    return tokens, positions


def simple_tokenizer(sentence_string, separator_re):

    tokens = []
    positions = []
    start = 0
    for point in separator_re.finditer(sentence_string):

        end = point.start()
        token = sentence_string[start:end]
        separator = sentence_string[end:point.end()]

        # Add token if not empty
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

        # Add separator
        if separator.strip():
            tokens.append(separator)
            positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        token = sentence_string[start:end]
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

    return tokens, positions


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
