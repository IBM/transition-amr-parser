import re
import json
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import Counter
from transition_amr_parser.amr import AMR
from ipdb import set_trace


def read_amr(file_path, ibm_format=False, generate=False):
    if generate:
        # yields each AMr, faster but non sequential
        return amr_generator(file_path, ibm_format=ibm_format)
    else:
        return amr_iterator(file_path, ibm_format=ibm_format)


def amr_iterator(file_path, ibm_format=False):
    '''
    Read AMRs in PENMAN+ISI-alignments or JAMR+alignments (ibm_format=True)

    (tokenize is deprecated)
    '''

    amrs = []
    # loop over blocks separated by whitespace
    tqdm_iterator = read_blocks(file_path)
    num_amr = len(tqdm_iterator)
    for index, raw_amr in enumerate(tqdm_iterator):

        if ibm_format:
            # From JAMR plus IBMs alignment format (DEPRECATED)
            amrs.append(AMR.from_metadata(raw_amr))
        else:
            # from penman
            amrs.append(AMR.from_penman(raw_amr))

        tqdm_iterator.set_description(f'Reading AMRs {index+1}/{num_amr}')

    return amrs


def amr_generator(file_path, ibm_format=False):
    '''
    Read AMRs in PENMAN+ISI-alignments or JAMR+alignments (ibm_format=True)

    (tokenize is deprecated)
    '''

    # loop over blocks separated by whitespace
    tqdm_iterator = read_blocks(file_path)
    num_amr = len(tqdm_iterator)
    for index, raw_amr in enumerate(tqdm_iterator):

        if ibm_format:
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
        return tqdm(blocks)
    else:
        return tqdm(blocks)


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
