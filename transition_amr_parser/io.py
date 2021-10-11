import re
import json
from collections import Counter, defaultdict
import subprocess
import ast
import xml.etree.ElementTree as ET
import penman
from tqdm import tqdm
from penman.layout import Push


def protected_tokenizer(sentence_string):
    separator_re = re.compile(r'[\.,;:?!"\' \(\)\[\]\{\}]')
    return simple_tokenizer(sentence_string, separator_re)


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


def get_amr(sentences, penmans):
    graph = penman.decode(penmans)
    nodes, edges = get_simple_graph(graph)
    # get tokens
    # TODO: penman module should read this as well
    tokens = sentences.split()
    return AMR(tokens, nodes, edges, penman=graph)


def get_simple_graph(graph):
    """
    Get simple nodes/edges representation from penman class
    """

    # get map of node variables to node names (this excludes constants)
    name_to_node = {x.source: x.target for x in graph.instances()}

    # Get all edges (excludes constants)
    edges = []
    for x in graph.edges():
        assert x.target in name_to_node
        edge_epidata = graph.epidata[(x.source, x.role, x.target)]
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            edges.append((x.target, f'{x.role}-of', x.source))
        else:
            edges.append((x.source, x.role, x.target))

    # Add constants both to node map and edges, use position in attribute as id
    for index, att in enumerate(graph.attributes()):
        assert index not in name_to_node
        name_to_node[index] = att.target
        edge_epidata = graph.epidata[(att.source, att.role, att.target)]
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            raise Exception()
            edges.append((index, f'{att.role}-of', att.source))
        else:
            edges.append((att.source, att.role, index))

    # print(penman.encode(graph))
    return name_to_node, edges


def legacy_graph_printer(metadata, nodes, root, edges):

    # These symbols can not be used directly for nodes
    must_scape_symbols = [':', '/', '(', ')']

    # start from meta-data
    output = metadata

    # identify nodes that should be quoted
    # find leaf nodes
    non_leaf_ids = set()
    for (src, label, trg) in edges:
        non_leaf_ids.add(src)
    leaf_ids = set(nodes.keys()) - non_leaf_ids
    # Find leaf nodes at end of :op or numeric ones
    quoted_nodes = []
    for (src, label, trg) in edges:
        if trg not in leaf_ids:
            continue
        if (
            nodes[src] == 'name'
            and re.match(r':op[0-9]+', label.split('-')[0])
        ):
            # NE Elements
            quoted_nodes.append(trg)
        elif any(s in nodes[trg] for s in must_scape_symbols):
            # Special symbols
            quoted_nodes.append(trg)
    # Add quotes to those
    for nid in quoted_nodes:
        if '"' not in nodes[nid]:
            nodes[nid] = f'"{nodes[nid]}"'

    # Determine short name for variables
    new_ids = {}
    for n in nodes:
        new_id = nodes[n][0] if nodes[n] else 'x'
        if new_id.isalpha() and new_id.islower():
            if new_id in new_ids.values():
                j = 2
                while f'{new_id}{j}' in new_ids.values():
                    j += 1
                new_id = f'{new_id}{j}'
        else:
            j = 0
            while f'x{j}' in new_ids.values():
                j += 1
            new_id = f'x{j}'
        new_ids[n] = new_id
    depth = 1
    out_nodes = {root}
    completed = set()

    # if 0 in nodes and nodes[0] == 'expressive':
    #     breakpoint()

    # Iteratively replace wildcards in this string to create penman notation
    amr_string = f'[[{root}]]'
    while '[[' in amr_string:
        tab = '      '*depth
        for n in out_nodes.copy():
            id = new_ids[n] if n in new_ids else 'r91'
            concept = nodes[n] if n in new_ids and nodes[n] else 'None'
            out_edges = sorted([e for e in edges if e[0] == n],
                               key=lambda x: x[1])
            targets = set(t for s, r, t in out_edges)
            out_edges = [f'{r} [[{t}]]' for s, r, t in out_edges]
            children = f'\n{tab}'.join(out_edges)
            if children:
                children = f'\n{tab}'+children
            if n not in completed:
                if (
                    concept[0].isalpha()
                    and concept not in [
                        'imperative', 'expressive', 'interrogative'
                    ]
                    # TODO: Exception :era AD
                    and concept != 'AD'
                ) or targets or (
                    # NOTE corner case: no child nodes, no parents either -> just a single node (otherwise the graph
                    #                   will not be connected)
                    concept in ['imperative', 'expressive', 'interrogative', 'AD']
                    and len(nodes) == 1        # TODO handle the corner case better
                ):
                    amr_string = amr_string.replace(
                        f'[[{n}]]', f'({id} / {concept}{children})', 1)
                else:
                    amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
                    # TODO does this affect Smatch? Yes it does affect Smatch...
                    # amr_string = amr_string.replace(
                    #     f'[[{n}]]', f'({id} / {concept}{children})', 1)
                completed.add(n)
            amr_string = amr_string.replace(f'[[{n}]]', f'{id}')
            out_nodes.remove(n)
            out_nodes.update(targets)
        depth += 1

    # sanity checks
    if len(completed) < len(out_nodes):
        raise Exception("Tried to print an uncompleted AMR")
    if (
        amr_string.startswith('"')
        or amr_string[0].isdigit()
        or amr_string[0] == '-'
    ):
        amr_string = '(x / '+amr_string+')'
    if not amr_string.startswith('('):
        amr_string = '('+amr_string+')'
    if len(nodes) == 0:
        amr_string = '(a / amr-empty)'
    elif len(nodes) == 1 and '/' not in amr_string:
        # FIXME: bad method to detect a constant as single node
        amr_string = '(a / amr-empty)'

    output += amr_string + '\n\n'

    return output


default_rel = ':rel'


class AMR():

    def __init__(self, tokens, nodes, edges, root, penman=None,
                 alignments=None, clean=True, connect=False, id=None):

        # make graph un editable
        self.tokens = tokens
        self.nodes = nodes
        self.edges = edges
        self.penman = penman
        self.alignments = alignments
        self.id = id

        # edges by parent
        self.edges_by_parent = defaultdict(list)
        for (source, edge_name, target) in edges:
            self.edges_by_parent[source].append((target, edge_name))

        # edges by child
        self.edges_by_child = defaultdict(list)
        for (source, edge_name, target) in edges:
            self.edges_by_child[target].append((source, edge_name))

        # root
        self.root = root

        # do the cleaning when necessary (e.g. build the AMR graph from model output, which might not be valid)
        if clean:
            self.clean_amr()    # cleaning is needed for oracle for AMR 3.0 training data
        if connect:
            self.connect_graph()

        # if self.root is None:
        #     # breakpoint()
        #     self.connect_graph()

        # breakpoint()
        # if self.tokens[0] == 'Haha':
        #     breakpoint()

    def clean_amr(self):
        # empty graph
        if not self.nodes:
            # breakpoint()
            # randomly add a single node
            for tok in self.tokens:
                if tok not in ['(', ')', ':', '"', "'", '/', '\\', '.', '?', '!', ',', ';']:
                    self.nodes[0] = tok
                    break
            if not self.nodes:
                self.nodes[0] = 'amr-empty'

            self.root = 0

        # clean concepts
        for n in self.nodes:
            if self.nodes[n] in ['.', '?', '!', ',', ';', '"', "'"]:
                self.nodes[n] = 'PUNCT'
            if self.nodes[n].startswith('"') and self.nodes[n].endswith('"'):
                self.nodes[n] = '"' + self.nodes[n].replace('"', '') + '"'
            if not (self.nodes[n].startswith('"') and self.nodes[n].endswith('"')):
                for ch in ['/', ':', '(', ')', '\\']:
                    if ch in self.nodes[n]:
                        self.nodes[n] = self.nodes[n].replace(ch, '-')
            if not self.nodes[n]:
                self.nodes[n] = 'None'
            if ',' in self.nodes[n]:
                self.nodes[n] = '"' + self.nodes[n].replace('"', '') + '"'
            if not self.nodes[n][0].isalpha() and not self.nodes[n][0].isdigit(
            ) and not self.nodes[n][0] in ['-', '+']:
                self.nodes[n] = '"' + self.nodes[n].replace('"', '') + '"'

        # clean edges
        for j, e in enumerate(self.edges):
            s, r, t = e
            if not r.startswith(':'):
                r = ':' + r
            e = (s, r, t)
            self.edges[j] = e

        # handle missing nodes (this shouldn't happen but a bad sequence of actions can produce it)
        for s, r, t in self.edges:
            if s not in self.nodes:
                # breakpoint()
                self.nodes[s] = 'NA'
            if t not in self.nodes:
                # breakpoint()
                self.nodes[t] = 'NA'

    def connect_graph(self):
        assigned_root = None

        # ========== this deals with the special structure where a dummy root node is marked with id -1,
        # i.e. self.nodes[-1] = 'root' (legacy; should not be the case for oracle 10)
        # here we remove the dummy root node to have a uniform representation of the graph attributes
        root_edges = []
        if -1 in self.nodes:
            del self.nodes[-1]

        for s, r, t in self.edges:
            if s == -1 and r == "root":
                assigned_root = t
            if s == -1 or t == -1:
                root_edges.append((s, r, t))
        for e in root_edges:
            self.edges.remove(e)
        # ==============================================================================================

        assert self.nodes, 'the graph should not be empty'

        assigned_root = self.root

        # ===== find all the descendants for each node
        descendents = {n: {n} for n in self.nodes}

        for x, r, y in self.edges:
            descendents[x].update(descendents[y])
            for n in descendents:
                if x in descendents[n]:
                    descendents[n].update(descendents[x])

        # all the ascendants for each node (including the node itself)
        ascendants = {n: {n} for n in self.nodes}
        for p, ds in descendents.items():
            for x in ds:
                ascendants[x].add(p)

        # ===== remove nodes that should not be potential root
        # - nodes with a parent (OR any ascendant)  && the parent/ascendant is not a descendant of the node
        #   (cycling case, not strictly a DAG, but this appears in AMR)
        # - nodes with no children

        potential_roots = [n for n in self.nodes]

        for n in potential_roots.copy():
            for p, ds in descendents.items():
                if n in ds and p not in descendents[n]:
                    potential_roots.remove(n)
                    break
            else:
                # above case not found
                if len(descendents[n]) == 1:
                    # only one descendent is itself, i.e. no children
                    potential_roots.remove(n)

        # ===== assign root (give priority to "multi-sentence" (although it could be non-root) or assigned_root)
        if potential_roots:
            # # pick the root with most descendents
            # potential_roots_nds = [len(descendents[r]) for r in potential_roots]
            # self.root = max(zip(potential_roots, potential_roots_nds), key=lambda x: x[1])[0]
            # # pick the root with bias towards earlier nodes
            self.root = potential_roots[0]
            for n in potential_roots:
                if self.nodes[n] == 'multi-sentence' or n == assigned_root:
                    self.root = n
        else:
            self.root = max(self.nodes.keys(),
                            key=lambda x: len([e for e in self.edges if e[0] == x])
                            - len([e for e in self.edges if e[2] == x]))

        # ===== connect graph
        # find disconnected nodes: only those disconnected roots of subgraphs
        disconnected = []

        for n in self.nodes:
            if self.root in ascendants[n]:
                # any node reachable by the root -> not disconnected
                continue

            if len(ascendants[n]) == 1:
                # only ascendant is itself, i.e. no parent
                disconnected.append(n)
            else:
                for p in ascendants[n]:
                    if p not in descendents[n]:
                        # there is any parent that is not in a cycle -> don't add (not a root of any subgraph)
                        break
                else:
                    # all the parents are current node's children: cycle -> only add if no node in cycle already added
                    if not any([m in ascendants[n] for m in disconnected]):
                        disconnected.append(n)

        if len(disconnected) > 0:
            for n in disconnected:
                self.edges.append((self.root, default_rel, n))

        # debug for amr2.0 training data
        # if 'From among them' in ' '.join(self.tokens):
        #     breakpoint()

    @classmethod
    def from_penman(cls, penman_text, tokenize=False):
        """
        Read AMR from penman notation (will ignore graph data in metadata)
        """
        graph = penman.decode(penman_text)
        nodes, edges = get_simple_graph(graph)
        if tokenize:
            assert 'snt' in graph.metadata, "AMR must contain field ::snt"
            tokens, _ = protected_tokenizer(graph.metadata['snt'])
        else:
            assert 'tok' in graph.metadata, "AMR must contain field ::tok " \
                "(or call this with tokenize=True)"
            tokens = graph.metadata['tok'].split()

        graph_id = None
        if 'id' in graph.metadata:
            graph_id = graph.metadata['id']

        return cls(tokens, nodes, edges, graph.top, penman=graph, clean=True,
                   connect=False, id=graph_id)

    @classmethod
    def from_metadata(cls, penman_text, tokenize=False):
        """Read AMR from metadata (IBM style)"""

        # Read metadata from penman
        field_key = re.compile(f'::[A-Za-z]+')
        metadata = defaultdict(list)
        separator = None
        for line in penman_text:
            if line.startswith('#'):
                line = line[2:].strip()
                start = 0
                for point in field_key.finditer(line):
                    end = point.start()
                    value = line[start:end]
                    if value:
                        metadata[separator].append(value)
                    separator = line[end:point.end()][2:]
                    start = point.end()
                value = line[start:]
                if value:
                    metadata[separator].append(value)

        assert 'tok' in metadata, "AMR must contain field ::tok"
        if tokenize:
            assert 'snt' in metadata, "AMR must contain field ::snt"
            tokens, _ = protected_tokenizer(metadata['snt'])
        else:
            assert 'tok' in metadata, "AMR must contain field ::tok"
            assert len(metadata['tok']) == 1
            tokens = metadata['tok'][0].split()
        nodes = {}
        alignments = {}
        edges = []
        for key, value in metadata.items():
            if key == 'edge':
                for items in value:
                    items = items.split('\t')
                    if len(items) == 6:
                        _, _, label, _, src, tgt = items
                        edges.append((src, f':{label}', tgt))
            elif key == 'node':
                for items in value:
                    items = items.split('\t')
                    if len(items) > 3:
                        _, node_id, node_name, alignment_str = items
                        start, end = alignment_str.split('-')
                        indices = list(range(int(start), int(end)))
                        alignments[node_id] = indices
                    else:
                        _, node_id, node_name = items
                        alignments[node_id] = None
                    nodes[node_id] = node_name
            elif key == 'root':
                root = value[0].split('\t')[1]

        # read metadata
        graph_id = None
        if metadata['id']:
            graph_id = metadata['id'][0].strip()

        return cls(tokens, nodes, edges, root, penman=None,
                   alignments=alignments, clean=True, connect=False,
                   id=graph_id)

    def get_metadata(self):
        """
        Returns graph information in the meta-data
        """
        assert self.root is not None, "Graph must be complete"
        output = ''
        output += '# ::tok ' + (' '.join(self.tokens)) + '\n'
        for n in self.nodes:
            alignment = ''
            if n in self.alignments and self.alignments[n] is not None:
                if type(self.alignments[n]) == int:
                    start = self.alignments[n]
                    end = self.alignments[n] + 1
                    alignment = f'\t{start}-{end}'
                else:
                    alignments_in_order = sorted(list(self.alignments[n]))
                    start = alignments_in_order[0]
                    end = alignments_in_order[-1] + 1
                    alignment = f'\t{start}-{end}'

            nodes = self.nodes[n] if n in self.nodes else "None"
            output += f'# ::node\t{n}\t{nodes}' + alignment + '\n'
        # root
        roots = self.nodes[self.root] if self.root in self.nodes else "None"
        output += f'# ::root\t{self.root}\t{roots}\n'
        # edges
        for s, r, t in self.edges:
            r = r.replace(':', '')
            edges = self.nodes[s] if s in self.nodes else "None"
            nodes = self.nodes[t] if t in self.nodes else "None"
            output += f'# ::edge\t{edges}\t{r}\t' \
                      f'{nodes}\t{s}\t{t}\t\n'
        return output

    def __str__(self):

        if self.penman:
            return penman.encode(self.penman)
        else:
            return legacy_graph_printer(self.get_metadata(), self.nodes,
                                        self.root, self.edges)

    def parents(self, node_id):
        return self.edges_by_child.get(node_id, [])

    def children(self, node_id):
        return self.edges_by_parent.get(node_id, [])

    def toJAMRString(self):
        """
        FIXME: Just modifies ::node line with respect to the original
        """
        output = penman.encode(self.penman)
        # Try first to just modify existing JAMR annotation
        new_lines = []
        modified = False
        for line in output.split('\n'):
            if line.startswith('# ::node'):
                modified = True
                items = line.split('\t')
                node_id = items[1]
                if node_id in self.alignments:
                    start = min(self.alignments[node_id])
                    dend = max(self.alignments[node_id]) + 1
                    if len(items) == 4:
                        items[-1] = f'{start}-{dend}'
                    elif len(items) == 3:
                        items.append(f'{start}-{dend}')
                    else:
                        raise Exception()
                line = '\t'.join(items)
            new_lines.append(line)
        # if not we write it ourselves
        if not modified:
            from ipdb import set_trace
            set_trace(context=30)
            print()
        return ('\n'.join(new_lines)) + '\n'


def read_amr2(file_path, ibm_format=False, tokenize=False, bar=True):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = []
        if bar:
            bar = tqdm
        else:
            def bar(x, desc=None): return x
        for line in bar(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                if ibm_format:
                    # From ::node, ::edge etc
                    raw_amrs.append(
                        AMR.from_metadata(raw_amr, tokenize=tokenize)
                    )
                else:
                    # From penman
                    raw_amrs.append(
                        AMR.from_penman(raw_amr, tokenize=tokenize)
                    )
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs


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
