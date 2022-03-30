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


class AMR():

    # relation used for detached subgraph
    default_rel = ':rel'
    # these need to be scaped in node names
    reserved_amr_chars = [':', '/', '(', ')']

    def __init__(self, tokens, nodes, edges, root, penman=None,
                 alignments=None, sentence=None, clean=True, connect=False,
                 id=None):

        # make graph un editable
        self.sentence = str(sentence)
        self.tokens = list(tokens)
        self.nodes = dict(nodes)
        self.edges = list(edges)
        self.penman = penman
        self.alignments = dict(alignments) if alignments else None
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

        # do the cleaning when necessary (e.g. build the AMR graph from model
        # output, which might not be valid)
        # if clean:
            # cleaning is needed for oracle for AMR 3.0 training data
        #    self.clean_amr()
        # if connect:
        #    self.connect_graph()

        # locate all heads of the entire graph (there should be only one)
        heads = [n for n in self.nodes if len(self.parents(n)) == 0]
        # heuristics to find the root
        if self.root is None:
            if len(heads) == 1:
                self.root = heads[0]
            elif 'multi-sentence' in self.nodes.values():
                for nid in heads:
                    if 'multi-sentence' == self.nodes[nid]:
                        self.root = nid
                        break
            else:
                # heuristic to find a root if missing
                # simple criteria, head with more children is the root
                self.root= sorted(heads, key=lambda n: len(self.children(n)))[-1]

        if len(heads) > 1:
            # add rel edges from detached subgraphs to the root
            heads.remove(self.root)
            for n in heads:
                self.edges.append((self.root, AMR.default_rel, n))

        # redo edges
        # edges by parent
        self.edges_by_parent = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self.edges_by_parent[source].append((target, edge_name))

        # edges by child
        self.edges_by_child = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self.edges_by_child[target].append((source, edge_name))

    def clean_amr(self):
        # empty graph
        if not self.nodes:
            # breakpoint()
            # randomly add a single node
            for tok in self.tokens:
                if tok not in [
                    '(', ')', ':', '"', "'", '/', '\\', '.', '?', '!', ',', ';'
                ]:
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
            if not (
                self.nodes[n].startswith('"') and self.nodes[n].endswith('"')
            ):
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

        # handle missing nodes (this shouldn't happen but a bad sequence of
        # actions can produce it)
        for s, r, t in self.edges:
            if s not in self.nodes:
                # breakpoint()
                self.nodes[s] = 'NA'
            if t not in self.nodes:
                # breakpoint()
                self.nodes[t] = 'NA'

    def connect_graph(self):
        assigned_root = None

        # this deals with the special structure where a dummy root node is
        # marked with id -1, i.e. self.nodes[-1] = 'root' (legacy; should not
        # be the case for oracle 10) here we remove the dummy root node to have
        # a uniform representation of the graph attributes
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
        #

        assert self.nodes, 'the graph should not be empty'

        assigned_root = self.root

        # find all the descendants for each node
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

        # remove nodes that should not be potential root
        # - nodes with a parent (OR any ascendant)  && the parent/ascendant is
        #   not a descendant of the node (cycling case, not strictly a DAG, but
        #   this appears in AMR)
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

        # assign root (give priority to "multi-sentence" (although it could be
        # non-root) or assigned_root)
        if potential_roots:
            # # pick the root with most descendents
            # potential_roots_nds = [len(descendents[r]) for r in
            # potential_roots]
            # self.root = max(zip(potential_roots, potential_roots_nds),
            # key=lambda x: x[1])[0]
            # # pick the root with bias towards earlier nodes
            self.root = potential_roots[0]
            for n in potential_roots:
                if self.nodes[n] == 'multi-sentence' or n == assigned_root:
                    self.root = n
        else:
            self.root = max(
                self.nodes.keys(),
                key=lambda x: len([e for e in self.edges if e[0] == x])
                - len([e for e in self.edges if e[2] == x])
            )

        # connect graph
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
                        # there is any parent that is not in a cycle -> don't
                        # add (not a root of any subgraph)
                        break
                else:
                    # all the parents are current node's children: cycle ->
                    # only add if no node in cycle already added
                    if not any([m in ascendants[n] for m in disconnected]):
                        disconnected.append(n)

        if len(disconnected) > 0:
            for n in disconnected:
                self.edges.append((self.root, AMR.default_rel, n))

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
        return cls(tokens, nodes, edges, graph.top, penman=graph,
                   alignments=alignments, sentence=sentence, clean=True,
                   connect=False, id=graph_id)

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
                   alignments=alignments, sentence=sentence, clean=True,
                   connect=False, id=graph_id)

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
                  is_constant=None):

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
            self.get_metadata(), self.nodes, self.root, self.edges
        )

    def parents(self, node_id):
        return self.edges_by_child.get(node_id, [])

    def children(self, node_id):
        return self.edges_by_parent.get(node_id, [])


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
    attributes = sorted(graph.attributes(),
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
