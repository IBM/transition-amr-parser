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

from collections import defaultdict
import re
from copy import deepcopy
# need to be installed with pip install penman
import penman
from penman.layout import Push
from penman.graph import Graph, Edge, Attribute
from penman import surface
from ipdb import set_trace
from transition_amr_parser.clbar import yellow_font


alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')


def normalize(token):
    """
    Normalize token or node
    """
    if token == '"':
        return token
    else:
        return token.replace('"', '')


def create_valid_amr(tokens, nodes, edges, root, alignments):

    # edges by parent
    parents = defaultdict(list)
    for (source, edge_name, target) in edges:
        parents[source].append((target, edge_name))

    # edges by child
    children = defaultdict(list)
    for (source, edge_name, target) in edges:
        children[target].append((source, edge_name))

    # locate all heads of the entire graph (there should be only one)
    heads = [n for n in nodes if len(parents[n]) == 0]

    # heuristics to find the root
    if root is None:
        if len(heads) == 1:
            root = heads[0]
        elif 'multi-sentence' in nodes.values():
            for nid in heads:
                if 'multi-sentence' == nodes[nid]:
                    root = nid
                    break
        elif heads:
            # FIXME: this need for an if signals degenerate cases not being
            # captured
            # heuristic to find a root if missing
            # simple criteria, head with more children is the root
            root = sorted(
                heads, key=lambda n: len(children[n])
            )[-1]

    # heuristics join disconnected graphs
    if len(heads) > 1:
        # add rel edges from detached subgraphs to the root
        if root in heads:
            # FIXME: this should not be happening
            heads.remove(root)
        for n in heads:
            edges.append((root, AMR.default_rel, n))

    return tokens, nodes, edges, root, alignments


class AMR():

    # relation used for detached subgraph
    default_rel = ':rel'
    # these need to be scaped in node names
    reserved_amr_chars = [':', '/', '(', ')']

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

        # precompute results for parents() and children()
        self._cache_key = None
        self.cache_graph()

        # root
        self.root = root

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

    @classmethod
    def from_penman(cls, penman_text, tokenize=False):
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
            for okey in ['node', 'edge', 'root', 'short', 'alignments']:
                if key.startswith(okey):
                    delete_keys.append(key)
        for key in delete_keys:
            del graph.metadata[key]

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

        # if the AMR did not come from penman parsing e.g. from state machine
        # or JAMR. We redo the node ids
        # TODO: Maybe there is a clearer way to have this
        if self.penman is None:
            node_map = self.get_node_id_map()
        else:
            node_map = None

        return f'{self.to_penman(node_map=node_map)}\n\n'

    def get_node_id_map(self):
        ''' Redo the ids of a graph to ensure they are valid '''

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
                new_id = nname[0].lower()
                if new_id == '"':
                    new_id = nname[1].lower()
                repeat = 2
                while new_id in id_map.values():
                    char = nname[0].lower()
                    if char == '"':
                        char = nname[1].lower()
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
        const_symbols = ['-', '+']

        if self.children(nid):
            # if it has children, it is not a constant
            return False
        elif (
            numeric_regex.match(self.nodes[nid])
            or self.nodes[nid] in const_symbols
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

    def scape_node_names(self):
        '''
        heuristic to decide if a node name must be scaped with quotes
        '''
        op_regex = re.compile(':op[0-9]+')
        numeric_regex = re.compile(r'["0-9,\.:-]')
        nodes = {}
        for nid, nname in self.nodes.items():
            if nname[0] == '"' and nname[-1] == '"':
                # has already quotes, ignore
                new_nname = nname
            elif any(c in nname for c in AMR.reserved_amr_chars):
                # has invalid characters, must scape
                new_nname = f'"{nname}"'
            elif (
                any(e == ':wiki' for _, e in self.parents(nid))
                or any(
                    op_regex.match(e) and self.nodes[n] == 'name'
                    for n, e in self.parents(nid)
                )
            ):
                # is NER leaf or child of wiki
                new_nname = f'"{nname}"'

            elif self.is_constant(nid) and not numeric_regex.match(nname):
                # it is a non-numeric constant
                new_nname = f'"{nname}"'

            else:
                # rest of cases, do not scape
                new_nname = nname
            nodes[nid] = new_nname
        self.nodes = nodes

    def to_penman(self, isi=True, node_filter=None, node_map=None,
                  is_constant=None, metadata=None):

        # make a copy to avoid modifying
        amr = deepcopy(self)

        # identify constant nodes by rules if no values provided
        if is_constant is None:
            is_constant = [n for n in amr.nodes if amr.is_constant(n)]

        # scape invalid node names
        amr.scape_node_names()

        # map ids if solicited
        if node_map is not None:
            amr.remap_ids(node_map)
            is_constant = [node_map[n] for n in is_constant]

        # metadata
        if metadata is None:
            metadata = {}
        if amr.sentence:
            metadata['snt'] = amr.sentence
        if amr.tokens:
            metadata['tok'] = ' '.join(amr.tokens)

        # extract basics
        # instances (non constant nodes)
        instances = [(amr.root, ':instance', amr.nodes[amr.root])]
        for nid, nname in amr.nodes.items():
            if nid not in is_constant and nid != amr.root:
                instances.append((nid, ':instance', nname))
        # edges and atributes (constant nodes)
        edges = []
        attributes = []
        for (src, label, trg) in amr.edges:
            if trg in is_constant:
                attributes.append(Attribute(src, label, amr.nodes[trg]))
            else:
                edges.append(Edge(src, label, trg))

        # intialize a graph
        g = Graph(instances + edges + attributes)
        g.metadata = metadata

        # port the alignmens from the AMR class to the penman module class
        g = add_alignments_to_penman(g, amr.alignments)

        return penman.encode(g, indent=4)

    def to_jamr(self):
        """ FIXME: do not use """
        return legacy_graph_printer(
            self.get_jamr_metadata(), self.nodes, self.root, self.edges
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
    if root is not None:
        # FIXME: This is allowed to complete the printing, but this graph is
        # already invalid.
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
            alignments[x.source] = list(isi_alignments[x].indices)

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
    for index, att in enumerate(attributes):
        assert index not in name_to_node
        # will be used as a node id, needs to be a string
        index = str(index)
        name_to_node[index] = att.target
        # add alignments
        if att in isi_alignments:
            alignments[index] = list(isi_alignments[att].indices)
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

    if root is None:
        # FIXME: This is to catch the if root is None: of get_jamr_metadata
        return '(a / amr-empty)\n\n'

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
