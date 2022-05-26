#    Copyright 2021 International Business Machines
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# This file is standalone and intended to be used as well separately of the
# repository, hence the attached license above.

from collections import defaultdict, Counter
import string
import re
# need to be installed with pip install penman
import penman
from penman.layout import Push
from penman import surface
from ipdb import set_trace
from transition_amr_parser.clbar import yellow_font


alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')


# known annotation issues in different corpora
ANNOTATION_ISSUES = {
    'amr2-train': [
        # repeated edge and child
        # :polarity
        2657,   # Each time the owners said "NO" don't worry about.
        # :mod
        17055,  # Oh boo hoo hoo!
        19264,  # Obviously, the reports of major downsizings by various ...
        # c :ARG1 g :ARG1-of c
        27279,  # I think Romney might have got the Catholic vote anyhow ...

        # "side" quoted and is not NER leaf (or a leaf in general)
        34110,

        # not attribute in string-entity : value
        # https://www.isi.edu/~ulf/amr/lib/amr-dict.html#string-entity
        # :value "racist"
        19409,  # But the word racist is so over-used and so frequently ...
        19410,  # So, once again, my understanding of the word racist ...
        19412,  # Now, what is your definition of the word, racist?
        19414,  # I do not know what her definition of racist is, but if ...
        # :value "maroon"
        19862,  # Here's the thing, my theory, you are either some little ...
        # :value "might"
        24480,  # The key word there is "might".
        # :value "commenting"
        24526,  # I should have said disrupting and not commenting.
        # :value "exciting"
        25906,  # I agree with you but don't think 'exciting' is ...
        # :value "deniability"
        26128,  # Might be worth googling the word "deniability".

        # not attribute in :timezone
        # https://www.isi.edu/~ulf/amr/lib/popup/timezone.html
        # :timezone (l / local)
        # should be  local-02, but not an attribute error, REMOVED
        # 27501  # eight o'clock in the morning local time
    ],

    'amr2-dev': [
        # repeated edge and child
        # m2 :ARG2 c6
        599  # Afghanistan does produce the opium which is the raw ...
    ],

    'amr3-train': [

        # AMR2.0 34110
        53032,

        # repeated edge and child
        2657,   # AMR2.0
        # repeated i :ARG0 c'
        19578,  # it is NOT designed to bring the insurance companies down
        21937,  # AMR2.0's 19264
        # repeated c :mod h
        27459,  # But hey, hey, who really gives a shit if a bunch of ...
        # repeated p :mod c
        29248,  # The British pipe was usually used as a wine measure, ...
        # repeated p5 :quant 10
        33047,  # A poll conducted by Ipsos/MORI for the Sun newspaper ...
        # repeated i2 :mod 2
        33187,  # Its going to be quite a show - and it is entirely,
        # repeated d :time "17:48"
        33645,  # Blair reappears on shortlist to head EU � By Tony Barber
        # repeated j2 :mod f
        34039,  # She always wore Christmas jumpers but not cheerful ...
        # repeates e :time s
        34161,   # When he slept with his ex the last time I somehow ...
        # AMR2.0 19264
        41560,
        # p :mod c
        54617,  # To ensure that the money reaches the Iraqi program , ...

        # :wiki "-" instead of :wiki -
        # from 24440 on this is widespread, ignored in test
    ],

    'amr3-dev': [
        # repeated edge and child
        # AMR2.0's 599
        953
    ]
}


def normalize(token):
    """
    Normalize token or node
    """
    if token == '"':
        return token
    else:
        return token.replace('"', '')


class NodeStackErrors():

    def __init__(self):
        self.visited_edges = []
        self.forbidden_edges = []
        self.visited_nodes = []
        self.offending_stacks = {
            'cycle': [],
            'mini-cycle': [],
            'reentrant': [],
            'mising_edges': []
        }

    def update(self, edge_stack, new_edges):

        # check for re-entrancies or cycles
        for new_edge in list(new_edges):

            self.visited_edges.append(new_edge)
            if edge_stack[-1][-1] not in self.visited_nodes:
                self.visited_nodes.append(edge_stack[-1][-1])

            child = new_edge[-1]
            if child in [e[0] for e in edge_stack]:
                # long cycle: do not follow as it will yield a loop
                # edge is still valid AMR
                # set_trace(context=30)
                self.forbidden_edges.append(new_edge)
                self.offending_stacks['cycle'].append(edge_stack + [new_edge])

            elif child == edge_stack[-1][0]:
                # mini cycle: parent child with edges in different directions
                # this is not valid AMR
                new_edges.remove(new_edge)
                self.offending_stacks['mini-cycle'].append(
                    edge_stack + [new_edge]
                )

            elif child in [e[0] for e in self.visited_edges]:
                # re-entrancy
                if new_edge not in self.offending_stacks['reentrant']:
                    # self.forbidden_edges.append(new_edge)
                    self.offending_stacks['reentrant'].append(new_edge)

    def close(self, edges):
        # get stats about missing nodes
        missing_edges = [e for e in edges if e not in self.visited_edges]
        if missing_edges:
            self.offending_stacks['mising_edges'] = missing_edges
        nodes = [nid for edge in edges for nid in [edge[0], edge[-1]]]
        visited_nodes = [
            nid for edge in self.visited_edges for nid in [edge[0], edge[-1]]
        ]


def trasverse(edges, root, downwards=True, reentrant=False):
    '''
    returns a list of stacks of edges when doing a depth first trasversal of a
    graph

    transverse graph comming from AMR annotation. It detects mini-loops (same
    node pair) and normal loops and removes them, it also identifies
    re-entrant subgraphs

    edges        list of (node_id, node_label, node_id)
    root         node_id in edges
    downwards    return edge when going down the tree/graph otherwise upwards
    reentrant    include reentrant nodes
    '''

    # edges by parent
    children_by_nid = defaultdict(list)
    for (source, edge_name, target) in edges:
        children_by_nid[source].append((source, edge_name, target))

    # will handle errors/ warnings for invalid AMR
    error = NodeStackErrors()

    # travel down through graph
    dfs_edges = []
    edge_stack = [(None, None, root)]
    visited_nodes = set()
    while edge_stack:

        if downwards:
            dfs_edges.append(list(edge_stack))

        # collect new children edges to put on top of the stack
        # do not add forbidden edges to the stack
        if not reentrant and edge_stack[-1][-1] in visited_nodes:
            new_edges = []
        else:
            new_edges = [
                e for e in children_by_nid[edge_stack[-1][-1]]
                if (
                    e not in error.forbidden_edges
                    # and (reentrant or e[-1] not in visited_nodes)
                )
            ]
            visited_nodes.add(edge_stack[-1][-1])
        new_edges = sort_edges(new_edges)[::-1]

        # update stats for warning/errors on AMR graph validity
        error.update(edge_stack, new_edges)

        if new_edges:
            # downwards
            edge_stack.extend(new_edges)

        else:

            # upwards, remove all edges until we find unvisited children
            while (
                len(edge_stack) > 1
                and edge_stack[-2][0] != edge_stack[-1][0]
            ):
                if not downwards:
                    dfs_edges.append(list(edge_stack))
                edge_stack.pop()
            # and the edge at the same level as those children
            if not downwards:
                dfs_edges.append(list(edge_stack))
            edge_stack.pop()

    # close and compute stats
    error.close(edges)

    return dfs_edges, error.offending_stacks


def sort_edges(edges):
    '''
    Sort edges for display in penman notation. NOTE: That AMR notations have
    inconsistent ordering so there is no perfect match
    '''

    numeric_edge = re.compile(':op([0-9]+)|:snt([0-9]+)')

    if edges == []:
        return edges

    elif all(numeric_edge.match(e[1]) for e in edges):
        # sort edge<number> edges by number
        def edge_number(edge):
            return [
                int(x) for x in numeric_edge.match(edge[1]).groups() if bool(x)
            ].pop()
        edges = sorted(edges, key=edge_number)
        return edges
    else:
        # sort edges consistently with AMR annotations
        edge_by_label = defaultdict(list)
        for e in edges:
            edge_by_label[e[1]].append(e)
        # sort top tier labels by this order
        top_tier = []
        for label in [':li', ':wiki', ':name', ':purpose', ':quant']:
            top_tier.extend(edge_by_label[label])
        # sort rest alphabetically
        rest_edges = [e for e in edges if e not in top_tier]
        rest_edges = sorted(rest_edges, key=lambda x: x[1])
        return top_tier + rest_edges


def scape_node_names(nodes, edges, is_attribute):

    # numbers are allways attributes
    numeric_regex = re.compile(r'["0-9,\.:-]')

    ner_ids = []
    name_ids = []
    wiki_ids = []
    value_ids = []
    for e in edges:
        if e[1] == ':name' and nodes[e[2]] == 'name':
            ner_ids.append(e[2])
        elif nodes[e[2]] == 'name':
            name_ids.append(e[2])
        elif e[1] == ':wiki':
            wiki_ids.append(e[2])
        elif e[1] == ':value':
            value_ids.append(e[2])

    ner_leaves = []
    name_leaves = []
    for e in edges:
        if e[0] in ner_ids:
            ner_leaves.append(e[2])
        elif e[0] in name_ids:
            name_leaves.append(e[2])

    isolated_scaped_chars = ['-', '+']
    new_nodes = {}
    for nid, nname in nodes.items():
        if nname == '"':
            # FIXME: This should be solved at machine level
            # just a single quote, invalid, put some dummy symbol
            nname = '_'
            # raise Exception('Quotes can not be a single AMR node')
        elif nname[0] == '"' and nname[-1] == '"':
            # already quoted, ensure no quotes inside
            nname[1:-1].replace('"', '')
        elif len(nname.split()) > 1:
            # multi-token expression
            nname = f'"{nname}"'
        elif any(c in nname for c in AMR.reserved_amr_chars):
            # reserved chars, need to be scaped
            nname = f'"{nname}"'
        elif nname in isolated_scaped_chars:
            # some chars, if they appear in isolation, need to be scaped
            nname = f'"{nname}"'

        # below here: just aesthetics

        elif nid in ner_leaves and not re.match('^[0-9]+$', nname):
            # numeric is not a 100% working criteria
            # unquoted ner leaves
            nname = f'"{nname}"'
        elif (
            is_attribute[nid]
            and nid in value_ids
            and not numeric_regex.match(nname)
        ):
            # non numeric attribute values
            nname = f'"{nname}"'
        elif is_attribute[nid] and nid in name_leaves:
            # attribute :name leaves
            nname = f'"{nname}"'
        elif nid in wiki_ids and nname != "-":
            # wiki
            # the "-" rule does not apply on AMR3 sometimes
            nname = f'"{nname}"'
        new_nodes[nid] = nname
    return new_nodes


def get_attribute_ids_by_node(nodes):
    # returns node ids that are attributes

    # attribute rules by node name
    # these symbols are allways attributes
    # numbers are allways attributes
    numeric_regex = re.compile(r'["0-9,\.:-]')
    # single letters are allways attributes
    single_letter_regex = re.compile(r'^[A-Z]$')
    attribute_ids = []
    for nid, nname in nodes.items():

        if numeric_regex.match(nodes[nid]):
            # digit or time
            attribute_ids.append(nid)

        elif nodes[nid] in ['-', '+']:
            # constants
            attribute_ids.append(nid)

        elif single_letter_regex.match(nodes[nid]):
            # single letter in caps, it also a constant
            attribute_ids.append(nid)

    return attribute_ids


def get_attribute_ids_by_edge(nodes, edges):
    # returns node ids that are attributes

    # named entity node
    ner_nids = [
        e[2] for e in edges if e[1] == ':name' and nodes[e[2]] == 'name'
    ]

    # attribute rules by edge
    # TODO: This may not capture all cases
    propbank_regex = re.compile(r'[a-z-]-[0-9]+')
    # edges whis child is allways an attribute (unless it has children)
    const_edges = [':mode', ':polite', ':wiki', ':li', ':era', ':value']
    # parents whos :value child is allways an attribute
    value_const_parents = [
        'url-entity', 'amr-unintelligible', 'word', 'email-address-entity',
        'emoticon', 'phone-number-entity'
    ]
    # option
    option_regex = re.compile(r':op[0-9]+')
    attribute_ids = []
    for (src, label, trg) in edges:

        if trg in attribute_ids:
            # ignore nodes assigned by other rules
            pass
        elif src in ner_nids:
            attribute_ids.append(trg)
        elif label in const_edges and nodes[trg] != 'amr-unknown':
            attribute_ids.append(trg)
        elif label == ':value' and nodes[src] in value_const_parents:
            # subset of :value edges
            attribute_ids.append(trg)
        elif (
            label == ':value'
            and nodes[src] == 'String-entity'
            and not propbank_regex.match(nodes[trg])
        ):
            # subset of string-entity :value edges
            # this is actually a bug on annotations
            attribute_ids.append(trg)
        elif label == ':timezone' and not nodes[trg].startswith('local'):
            # subset of :timezone
            attribute_ids.append(trg)
        elif label == ':era' and nodes[trg] in ['AD', 'BC']:
            # subset of :era
            attribute_ids.append(trg)
        elif bool(option_regex.match(label)) and nodes[src] == 'name':
            # name :op<number>
            attribute_ids.append(trg)
        elif (
            nodes[src] == 'score-on-scale-91'
            and label in [':ARG1', ':ARG3']
            and nodes[trg] != 'amr-unknown'
        ):
            attribute_ids.append(trg)
        elif (
            nodes[src] == 'street-address-91'
            and label in [':ARG6']
            and nodes[trg] != 'amr-unknown'
        ):
            attribute_ids.append(trg)

    return attribute_ids


def get_is_atribute(nodes, edges):
    '''
    heuristic to determine which nodes in a graph are attributes

    TODO: this is imperfect
    '''

    # main rule: attributes do not have children
    is_attribute = {}
    for (src, label, trg) in edges:
        is_attribute[src] = False

    # rules based on nodes
    for nid in get_attribute_ids_by_node(nodes):
        if nid not in is_attribute:
            is_attribute[nid] = True

    # rules based on edges
    for nid in get_attribute_ids_by_edge(nodes, edges):
        if nid not in is_attribute:
            is_attribute[nid] = True

    # rest are false
    for n in nodes:
        if n not in is_attribute:
            is_attribute[n] = False

    return is_attribute


def get_isi_str(alignments):
    if alignments is None:
        return ''
    start = min(alignments)
    end = max(alignments)
    if start + 1 == end or start == end:
        return f'~{start}'
    else:
        end += 1
        return f'~{start},{end}'


def simple_to_penman(nodes, edges, root, alignments=None, isi=True,
                     color=False):

    # quick exit
    if nodes == {}:
        return '(a / amr-empty)\n'

    # rules to determine if a node is an attribute or variable
    is_attribute = get_is_atribute(nodes, edges)

    # ensure node name are valid
    nodes = scape_node_names(nodes, edges, is_attribute)

    # depth first trasversal, return all edges involved in paths from root to
    # current leaf as well as detected re-entrancies or loop reentrancies this
    # will not stop at re-entrancies but will remove mini-loops and ignore
    # loops
    edge_stacks, offending_stacks = trasverse(edges, root, reentrant=False)

    # needed statistics
    loop_edges = [x[-1] for x in offending_stacks['cycle']]
    loop_nids = set([x[-1] for x in loop_edges])
    reentrant_edges = offending_stacks['reentrant']
    reentrant_nids = set([x[-1] for x in reentrant_edges])

    penman_str = ''
    prev_stack = None
    do_not_close = False
    visited_nodes = set()
    while edge_stacks:

        # for edge_stack in edge_stacks:
        edge_stack = edge_stacks.pop(0)

        # stats
        edge = edge_stack[-1]
        pad = '    ' * (len(set(x[0] for x in edge_stack)) - 1)
        parent, label, nid = edge

        # close parentheses when moving upwards in the tree
        if prev_stack is not None and len(prev_stack) > len(edge_stack):
            prev_child = None
            # if node was re-entrant it has no termination parenthesis
            if do_not_close:
                do_not_close = False
                prev_stack.pop()
            # move upwards in the tree one depth at at a time by popping from
            # the edge stack
            while prev_stack != edge_stack:
                _, _, child = prev_stack.pop()
                if child != prev_child:
                    penman_str += ')'
                prev_child = child

        if prev_stack is not None:
            penman_str += '\n'

        # color
        nid_str = nid
        if color:
            if nid in loop_nids:
                nid_str = "\033[91m%s\033[0m" % nid
            elif nid in reentrant_nids:
                nid_str = "\033[93m%s\033[0m" % nid
            if label == ':rel':
                label = "\033[93m%s\033[0m" % label

        # node conditions
        # special case
        if (
            # label in [':wiki', ':polarity', ':polite']
            nodes[nid] in ['"-"', '"+"']
        ):
            nname = normalize(nodes[nid])
        else:
            nname = nodes[nid]

        # print edge
        if label:
            if is_attribute[nid]:
                # attribute
                penman_str += f'{pad}{label} {nname}'
                if alignments and nid in alignments:
                    if isi:
                        penman_str += get_isi_str(alignments[nid])

                do_not_close = True

            elif nid in visited_nodes:
                # re-entrancy
                penman_str += f'{pad}{label} {nid_str}'
                do_not_close = True

            else:
                # normal node
                penman_str += f'{pad}{label} ({nid_str} / {nname}'
                if alignments and nid in alignments:
                    if isi:
                        penman_str += get_isi_str(alignments[nid])

        else:
            penman_str += f'({nid_str} / {nname}'
            if alignments and nid in alignments:
                if isi:
                    penman_str += get_isi_str(alignments[nid])

        prev_stack = edge_stack

        # add this node as visited
        visited_nodes.add(nid)

    # close AMR
    while bool(prev_stack):
        if do_not_close:
            do_not_close = False
            prev_stack.pop()
        prev_stack.pop()
        penman_str += ')'
        if prev_stack == []:
            penman_str += '\n'

    return penman_str


def find_roots(edges, root):

    num_parents = Counter()
    num_children = Counter()
    nodes = set()
    for (s, l, t) in edges:
        num_parents[t] += 1
        num_children[s] += 1
        nodes.add(s)
        nodes.add(t)

    heads = [n for n in nodes if num_parents[n] == 0]
    # if we have edges and no head, there is a cycle. Cut at the node with
    # highest number of children
    if root is None and heads == [] and edges != []:
        heads = [num_children.most_common(1)[0][0]]
    return heads


def force_rooted_connected_graph(nodes, edges, root):

    # locate all heads of the entire graph (there should be only one)
    heads = find_roots(edges, root)

    # for each head, find its descendant, ignore loops
    head_descendants = dict()
    total_visited = set()
    for head in heads:
        stacks, offending_stacks = trasverse(edges, head)
        descendants = set([s[-1][-1] for s in stacks if s[-1][-1] != head])
        if stacks and stacks[0]:
            descendants |= set(stacks[0][0])
        head_descendants[head] = descendants
        total_visited |= set(descendants)
    loose_nodes = set(nodes) - total_visited

    # find root strategy: multi-sentence or head with more descendants
    if root is None:
        if 'multi-sentence' in nodes.values():
            index = list(nodes.values()).index('multi-sentence')
            root = list(nodes.keys())[index]
        elif edges == []:
            heads = list(nodes.keys())
            if nodes:
                root = heads[0]
        else:
            def key(x): return len(x[1])
            root = sorted(head_descendants.items(), key=key)[-1][0]

    # connect graph: add rel edges from detached subgraphs to the root
    for n in heads + sorted(loose_nodes):
        if n != root:
            edges.append((root, AMR.default_rel, n))

    return root, edges


def create_valid_amr(tokens, nodes, edges, root, alignments):

    if root is None:
        print(yellow_font('WARNING: missing root'))

    # rooted and connected
    # NOTE: be careful not to overwrite edges or root
    droot = root if root is None else str(root)
    root, edges = force_rooted_connected_graph(nodes, list(edges), droot)
    if any(e[1] == AMR.default_rel for e in edges):
        print(yellow_font('WARNING: disconnected graphs'))

    # TODO: Unclear if necessary depending on printer
    # nodes, edges = prune_mini_cycles(nodes, edges, root)
    # if alignments:
    #     alignments = {nid: alignments[nid] for nid in nodes}

    return tokens, nodes, edges, root, alignments


class AMR():

    # relation used for detached subgraph
    default_rel = ':rel'
    # these need to be scaped in node names
    reserved_amr_chars = [':', '/', '(', ')', '~']
    # TODO: also - + in isolation

    def __init__(self, tokens, nodes, edges, root, penman=None,
                 alignments=None, sentence=None, id=None):

        # make graph uneditable
        self.sentence = str(sentence) if sentence is not None else None
        self.tokens = list(tokens) if tokens is not None else None
        self.nodes = dict(nodes)
        self.edges = list(edges)
        self.penman = penman
        self.alignments = dict(alignments) if alignments else None
        self.id = id

        # root
        self.root = root

        # precompute results for parents() and children()
        self._cache_key = None
        self.cache_graph()

    def cache_graph(self):
        '''
        Precompute edges indexed by parent or child
        '''

        # If the cache has not changed, no need to recompute
        if self._cache_key == tuple(self.edges):
            return

        # edges by parent
        self._edges_by_parent = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self._edges_by_parent[source].append((target, edge_name))
        # sort alphabetically by edge i.e. ARG1 before ARG2
        # TODO: double check if this needed
        _edges_by_parent2 = {}
        for parent, children in self._edges_by_parent.items():
            _edges_by_parent2[parent] = \
                sorted(children, key=lambda c: c[1])[::-1]
        self._edges_by_parent = _edges_by_parent2

        # edges by child
        self._edges_by_child = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self._edges_by_child[target].append((source, edge_name))

        # store a key to know when to recompute
        self._cache_key == tuple(self.edges)

    def parents(self, node_id, edges=True):
        self.cache_graph()
        arcs = self._edges_by_child.get(node_id, [])
        if edges:
            return arcs
        else:
            return [a[0] for a in arcs]

    def children(self, node_id, edges=True):
        self.cache_graph()
        arcs = self._edges_by_parent.get(node_id, [])
        if edges:
            return arcs
        else:
            return [a[0] for a in arcs]

    @classmethod
    def from_penman(cls, penman_text):
        """
        Read AMR from penman notation (will ignore graph data in metadata)
        """

        assert isinstance(penman_text, str), "Expected string with EOL"
        assert '\n' in penman_text, "Expected string with EOL"

        graph = penman.decode(penman_text.split('\n'))
        nodes, edges, alignments = get_simple_graph(graph)
        if 'tok' in graph.metadata:
            tokens = graph.metadata['tok'].split()
        else:
            tokens = None

        graph_id = None
        if 'id' in graph.metadata:
            graph_id = graph.metadata['id']

        if 'snt' in graph.metadata:
            sentence = graph.metadata['snt']
        else:
            sentence = None

        # wipe out JAMR notation from metadata since it can become inconsistent
        # also remove unsupported "alignments" field
        delete_keys = []
        for key, data in graph.metadata.items():
            for okey in ['node', 'edge', 'root', 'short']:
                if key.startswith(okey):
                    delete_keys.append(key)
        for key in delete_keys:
            del graph.metadata[key]

        # if an alignments field is specified, chech if this is in replacement
        # of ISI alignment annotation

        # remove quotes
        nodes = {nid: normalize(nname) for nid, nname in nodes.items()}

        return cls(tokens, nodes, edges, graph.top, penman=graph,
                   alignments=alignments, sentence=sentence, id=graph_id)

    @classmethod
    def from_metadata(cls, penman_text):
        """Read AMR from metadata (IBM style)"""

        assert isinstance(penman_text, str), "Expected string with EOL"
        assert '\n' in penman_text, "Expected string with EOL"

        # Read metadata from penman
        field_key = re.compile(f'::[A-Za-z]+')
        metadata = defaultdict(list)
        separator = None
        for line in penman_text.split('\n'):
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

        graph_id = metadata['id'] if 'id' in metadata else None

        # extract graph from meta-data
        nodes = {}
        alignments = {}
        edges = []
        tokens = None
        sentence = None
        root = None
        for key, value in metadata.items():
            if key == 'snt':
                sentence = metadata['snt'][0].strip()
            elif key == 'tok':
                tokens = metadata['tok'][0].split()
            elif key == 'edge':
                for items in value:
                    items = items.split('\t')
                    if len(items) == 6:
                        _, _, label, _, src, tgt = items
                        edges.append((src, f':{label}', tgt))
            elif key == 'node':
                for items in value:
                    items = items.split('\t')
                    if len(items) > 3:
                        _, node_id, node_name, alignment = items
                        start, end = alignment_regex.match(alignment).groups()
                        indices = list(range(int(start), int(end)))
                        alignments[node_id] = indices
                    else:
                        _, node_id, node_name = items
                        alignments[node_id] = None
                    nodes[node_id] = node_name
            elif key == 'root':
                root = value[0].split('\t')[1]

        # filter bad nodes (only in edges)
        new_edges = []
        for (s, label, t) in edges:
            if s in nodes and t in nodes:
                new_edges.append((s, label, t))
            else:
                print(yellow_font('WARNING: edge with extra node (ignored)'))
                print((s, label, t))
        edges = new_edges

        # sanity check: there was some JAMR
        assert bool(nodes), "JAMR notation seems empty"

        # remove quotes
        nodes = {nid: normalize(nname) for nid, nname in nodes.items()}

        return cls(tokens, nodes, edges, root, penman=None,
                   alignments=alignments, sentence=sentence, id=graph_id)

    def get_jamr_metadata(self, penman=False):
        """
        Returns graph information in the meta-data
        """
        return get_jamr_metadata(
            self.tokens, self.nodes, self.edges, self.root, self.alignments,
            penman=penman
        )

    def __str__(self):
        return self.to_penman()

    def get_node_id_map(self):
        ''' Redo the ids of a graph to ensure they are valid '''

        def get_valid_char(nname):
            candidates = str(nname)
            while candidates and candidates[0] not in string.ascii_lowercase:
                candidates = candidates[1:]
            if candidates:
                return candidates[0]
            else:
                return 'a'

        # determine a map from old to new ids. Importnat to sort constants in
        # the same way as from_penman code to get the same mapping
        num_constants = 0
        id_map = {}
        for nid, nname in sorted(self.nodes.items(), key=lambda x: x[1]):
            # parents = self.parents(nid)
            # children = self.children(nid)
            if self.is_constant(nid):
                # detect constants
                id_map[nid] = str(num_constants)
                num_constants += 1
            else:
                new_id = get_valid_char(nname)
                repeat = 2
                while new_id in id_map.values():
                    char = get_valid_char(nname)
                    new_id = f'{char}{repeat}'
                    repeat += 1
                id_map[nid] = new_id

        # constants are sorted by full
        return id_map

    def remap_ids(self, id_map):

        # rewrite nodes, edges and alignments
        self.edges = [
            (id_map[s], label, id_map[t]) for (s, label, t) in self.edges
        ]
        self.nodes = {id_map[nid]: nname for nid, nname in self.nodes.items()}
        if self.alignments:
            self.alignments = {
                id_map[nid]: alignments
                for nid, alignments in self.alignments.items()
            }
        if self.root is not None:
            self.root = id_map[self.root]

    def is_constant(self, nid):
        '''
        heuristic to determine if node is a constant

        TODO: this is imperfect
        '''
        if nid not in self.nodes:
            set_trace(context=30)
        const_edges = [':mode', ':polite', ':wiki', ':li', ':timezone']
        # parents whos :value child is allways constant
        value_const_parents = ['url-entity', 'amr-unintelligible']
        op_regex = re.compile(':op[0-9]+')
        numeric_regex = re.compile(r'["0-9,\.:-]')
        single_letter_regex = re.compile(r'^[A-Z]$')
        # TODO: This may not capture all cases
        propbank_regex = re.compile(r'[a-z-]-[0-9]+')
        const_nodes = ['-', '+']

        if self.children(nid):
            # if it has children, it is not a constant
            return False
        elif (
            numeric_regex.match(self.nodes[nid])
            or self.nodes[nid] in const_nodes
        ):
            # digit or time
            return True

        elif any(e in const_edges for _, e in self.parents(nid)):
            # parent edges of constants only
            return True

        elif any(
            e == ':value'
            and self.nodes[n] in value_const_parents
            for n, e in self.parents(nid)
        ):
            # URL entity or amr-unintelligible values are constants
            return True

        elif any(
            e == ':value'
            and self.nodes[n] == 'string-entity'
            # FIXME: Unreliable detection of propbank
            and not propbank_regex.match(self.nodes[nid])
            for n, e in self.parents(nid)
        ):
            # FIXME: string-entity seems to have no clear rule to decide
            # between constant and variable
            return True

        elif any(
            op_regex.match(e) and self.nodes[n] == 'name'
            for n, e in self.parents(nid)
        ):
            # Named entity
            # TODO: This only checks for name parent, not actual NER
            return True
        elif (
            any(e == ':era' for _, e in self.parents(nid))
            and self.nodes[nid] in ['AD', 'BC']
        ):
            # specific alphanumeric era values
            # TODO: This is not a robust strategy
            return True

        elif single_letter_regex.match(self.nodes[nid]):
            # single letter in caps, it also a constant
            return True

        else:
            return False

    def get_metadata(self, isi=True):

        # metadata
        if self.penman is None:
            metadata_str = ''
            if self.sentence:
                metadata_str += f'# ::snt {self.sentence}\n'
            if self.tokens:
                tokens_str = ' '.join(self.tokens)
                metadata_str += f'# ::tok {tokens_str}\n'
        else:
            metadata_str = '\n'.join([
                f'# ::{k} {v}' for k, v in self.penman.metadata.items()
            ])
            metadata_str += '\n'

        # until Smatch reads ISI, admit also adding it as metadata
        if not isi and self.alignments:
            alignment_str = ' '.join([
                f'{nid}~' + ','.join(map(str, als))
                for nid, als in self.alignments.items()
            ])
            metadata_str += f'# ::alignments {alignment_str}\n'

        return metadata_str

    def to_penman(self, isi=True, jamr=False):

        if jamr:
            penman_str = self.get_jamr_metadata()
        else:
            penman_str = self.get_metadata(isi)

        # if this does not come from reading with penman moduel, redo ids as
        # the could come from jamr or parsing
        if self.penman is None:
            self.remap_ids(self.get_node_id_map())

        return penman_str + simple_to_penman(
            self.nodes, self.edges, self.root, self.alignments, isi=isi
        )


def get_jamr_metadata(tokens, nodes, edges, root, alignments, penman=False):
    """
    Returns graph information in the meta-data

    tokens      list(str)
    nodes       dict(str/int: str)
    edges       list(tuple)
    root        str/int
    alignments  dict(str/int: str)
    """
    output = ''
    output += '# ::tok ' + (' '.join(tokens)) + '\n'
    for n in nodes:
        alignment = ''
        if n in alignments and alignments[n] is not None:
            if type(alignments[n]) == int:
                start = alignments[n]
                end = alignments[n] + 1
                alignment = f'\t{start}-{end}'
            else:
                alignments_in_order = sorted(list(alignments[n]))
                start = alignments_in_order[0]
                end = alignments_in_order[-1] + 1
                alignment = f'\t{start}-{end}'

        nodes_str = nodes[n] if n in nodes else "None"
        output += f'# ::node\t{n}\t{nodes_str}' + alignment + '\n'
    # root
    roots = nodes[root] if root in nodes else "None"
    output += f'# ::root\t{root}\t{roots}\n'
    # edges
    for s, r, t in edges:
        r = r.replace(':', '')
        edges_str = nodes[s] if s in nodes else "None"
        nodes_str = nodes[t] if t in nodes else "None"
        output += f'# ::edge\t{edges_str}\t{r}\t' \
                  f'{nodes_str}\t{s}\t{t}\t\n'

    # format in a way that can be added to the penman class metadata
    if penman:
        new_output = {}
        for item in output.split('\n'):
            pieces = item.replace('# ::', '').split(' ')
            if pieces == ['']:
                continue
            key = pieces[0]
            rest = ' '.join(pieces[1:])
            new_output[key] = rest
        output = new_output

    return output


def add_alignments_to_penman(g, alignments, string=False, strict=True):

    # FIXME: strict = True
    for (nid, label, trg) in g.attributes():
        if (
            (nid, label, trg) in g.epidata
            and (nid in alignments or not strict)
        ):
            g.epidata[(nid, label, trg)].append(
                surface.Alignment(tuple(alignments[nid]), prefix='')
            )

    for (nid, label, nname) in g.instances():
        if (
            (nid, label, nname) in g.epidata
            and (nid in alignments or not strict)
        ):
            g.epidata[(nid, label, nname)].append(
                surface.Alignment(tuple(alignments[nid]), prefix='')
            )

    if string:
        return penman.encode(g, indent=4) + '\n'
    else:
        return g


def escape_nodes(amr):

    # these symbols must be scaped with quotes, also NERs
    must_scape_symbols = [':', '/', '(', ')']
    ner_nids = []
    for (source, label, target) in amr.edges:
        if source == 'name' and label == ':name':
            for (edge, ner_nid) in amr.children(target):
                if re.match(':op[0-9]+', edge):
                    ner_nids.append(ner_nid)
    new_nodes = {}
    for nid, nname in amr.nodes.items():
        if nname[0] == '"' and nname[-1] == '"':
            new_nodes[nid] = nname
        elif any(c in nname for c in must_scape_symbols):
            new_nodes[nid] = f'"{nname}"'
        elif nid in ner_nids:
            new_nodes[nid] = f'"{nname}"'
        else:
            new_nodes[nid] = nname

    amr.nodes = new_nodes

    return amr


def get_simple_graph(graph):
    """
    Get simple nodes/edges/alignments representation from penman class
    """

    # alignments
    isi_alignments = surface.alignments(graph)

    # get map of node variables to node names (this excludes constants)
    name_to_node = {}
    alignments = {}
    for x in graph.instances():
        name_to_node[x.source] = x.target
        if x in isi_alignments:
            if len(isi_alignments[x].indices) == 1:
                alignments[x.source] = list(isi_alignments[x].indices)
            elif len(isi_alignments[x].indices) == 2:
                start = isi_alignments[x].indices[0]
                end = isi_alignments[x].indices[-1]
                alignments[x.source] = list(range(start, end))
            else:
                raise Exception('Unexpected ISI alignment format')

    # reentrancy
    reentrancies = [e for e, c in graph.reentrancies().items() if c > 1]

    # Get all edges (excludes constants)
    edges = []
    re_entrant = defaultdict(list)
    for x in graph.edges():
        assert x.target in name_to_node

        # get epidata
        edge_epidata = graph.epidata[(x.source, x.role, x.target)]

        # keep inverted edges
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            edge = (x.target, f'{x.role}-of', x.source)
        else:
            edge = (x.source, x.role, x.target)

        # delay adding an edge if its reentrant until we produce the original
        # edge
        # TODO: Does not work fully, remove?
        if (
            edge[-1] in reentrancies
            and edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == edge[-1]
            and re_entrant[edge[-1]] is not None
        ):
            # this is the original edge from a re-entrant series
            # append edge and all the re-entrancies
            edges.append(edge)
            edges.extend(re_entrant[edge[-1]])
            # block, since we do not need it any more
            re_entrant[edge[-1]] = None

        elif (
            edge[-1] in reentrancies
            and re_entrant[edge[-1]] is not None
        ):
            # append also edges rentrant to these
            re_entrant[edge[-1]].append(edge)

        else:
            # append edge
            edges.append(edge)

    # if nodes are re-entran to root, we will reach here with pending edges
    for nid, rest in re_entrant.items():
        if rest is not None:
            edges.extend(rest)

    if len(edges) < len(graph.edges()):
        set_trace(context=30)

    # Add constants both to node map and edges, use position in attribute as id
    # sort attributes by target for consistency in assignment
    attributes = sorted(
        graph.attributes(),
        key=lambda x: (
            x.source, x.role, x.target.replace('"', '')
        )
    )
    index = 0
    for att in attributes:
        assert index not in name_to_node
        # will be used as a node id, needs to be a string
        # watch for existing numbers used as ids
        while str(index) in name_to_node:
            index += 1
        name_to_node[str(index)] = att.target
        # add alignments
        if att in isi_alignments:
            if len(isi_alignments[att].indices) == 1:
                alignments[str(index)] = list(isi_alignments[att].indices)
            elif len(isi_alignments[att].indices) == 2:
                start = isi_alignments[att].indices[0]
                end = isi_alignments[att].indices[-1]
                alignments[str(index)] = list(range(start, end))
            else:
                raise Exception('Unexpected ISI alignment format')

        edge_epidata = graph.epidata[(att.source, att.role, att.target)]
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            raise Exception()
            edges.append((str(index), f'{att.role}-of', att.source))
        else:
            edges.append((att.source, att.role, str(index)))

        # increase index
        index += 1

    return name_to_node, edges, alignments


def get_node_id(nids, name, node):
    '''Get a valid node id'''
    nid = name[0]
    idx = 2
    while nid in nids:
        nid = f'{name[0]}{idx}'
        idx += 1

    nids[nid] = node

    return nid


def legacy_graph_printer(metadata, nodes, root, edges):
    '''
    Legacy printer from stack-LSTM, stack-Transformer and action-pointer
    '''

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

    # scape all node names with must-scape symbols
    for nid, nname in nodes.items():
        if any(s in nname for s in must_scape_symbols):
            # Special symbols
            quoted_nodes.append(nid)

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
                    # NOTE corner case: no child nodes, no parents either ->
                    # just a single node (otherwise the graph will not be
                    # connected)
                    concept in [
                        'imperative', 'expressive', 'interrogative', 'AD'
                    ]
                    and len(nodes) == 1
                ):
                    amr_string = amr_string.replace(
                        f'[[{n}]]', f'({id} / {concept}{children})', 1)
                else:
                    amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
                    # TODO This does affect Smatch.
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
    output += f'# ::short\t{str(new_ids)}\t\n'
    output += amr_string + '\n\n'

    return output


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


class InvalidAMRError(Exception):
    pass
