import sys
from collections import defaultdict
import re
# pip install
from tqdm import tqdm
from ipdb import set_trace
import penman
from penman.layout import Push


def read_amr(file_path, ibm_format=False, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = []
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
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


alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')


class AMR():

    def __init__(self, tokens, nodes, edges, root, penman=None,
                 alignments=None):

        # make graph un editable
        self.tokens = tokens
        self.nodes = nodes
        self.edges = edges
        self.penman = penman
        self.alignments = alignments

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
    def from_penman(cls, penman_text, tokenize=False):
        """
        Read AMR from penman notation (will ignore graph data in metadata)
        """
        graph = penman.decode(penman_text)
        nodes, edges = get_simple_graph(graph)
        if tokenize:
            assert 'snt' in graph.metadata, "AMR must contain field ::tok"
            tokens, _ = protected_tokenizer(graph.metadata['snt'])
        else:
            assert 'tok' in graph.metadata, "AMR must contain field ::tok"
            tokens = graph.metadata['tok'].split()
        return cls(tokens, nodes, edges, graph.top, penman=graph)

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

        return cls(tokens, nodes, edges, root, penman=None,
                   alignments=alignments)

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
            return ' '.join(self.tokens) + '\n\n' + penman.encode(self.penman)
        else:
            return legacy_graph_printer(self.get_metadata(), self.nodes,
                                        self.root, self.edges)

    def toJAMRString(self):
        """
        FIXME: Just modifies ::node line with respect to the original
        """
        output = penman.encode(self.penman)
        new_lines = []
        modified = False
        for line in output.split('\n'):
            if line.startswith('# ::node'):
                modified = True
                items = line.split('\t')
                node_id = items[1]
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
        assert modified
        return ('\n'.join(new_lines)) + '\n'


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
                ) or targets:
                    amr_string = amr_string.replace(
                        f'[[{n}]]', f'({id} / {concept}{children})', 1)
                else:
                    amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
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

    output += amr_string + '\n\n'

    return output


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


def get_reentrancy_edges(amr):

    def first_alignment(edge):
        '''Return left most token position aligned to a source node'''
        return min(amr.alignments.get(edge[0], [1000]))

    # Get re-entrancy edges
    originals = []
    reentrancy_edges = []
    for nid, nname in amr.nodes.items():
        parents = amr.parents(nid, edges=False)
        if len(parents) > 1:
            sorted_edges = sorted(parents, key=first_alignment)
            reentrancy_edges.extend(sorted_edges[1:])
            originals.append(sorted_edges[0])

    return reentrancy_edges, originals


def get_path_ids(amr):

    # get children by parent index, removing re-entrancies
    reentrancy_edges, _ = get_reentrancy_edges(amr)
    children_by_nid = defaultdict(list)
    for (src, label, tgt) in amr.edges:
        if tgt not in reentrancy_edges:
            children_by_nid[src].append(tgt)

    # Depth first search to extract all paths
    node_id = amr.root
    paths = []
    path = [node_id]
    while True:
        if children_by_nid[path[-1]]:
            # Get next children for this node, remove that child
            path.append(children_by_nid[path[-1]].pop())
        elif path[-1] == amr.root:
            # no more children available but this is the root, we are finished
            break
        else:
            # no more children available jump one level up
            new_path = path[:-1]
            paths.append(path)
            path = new_path

    return paths


def get_print_paths(amr, paths):

    def path2str(path):
        str_path = f'{amr.nodes[path[0]]}'
        for n in range(len(path) - 1):
            str_path += f' {edge_by_nids[tuple(path[n:n+2])]}'
            str_path += f' {amr.nodes[path[n+1]]}'
        return str_path

    edge_by_nids = {(src, tgt): label for (src, label, tgt) in amr.edges}
    print_paths = []
    for path in paths:
        if isinstance(path, tuple):
            subpaths = []
            for spath in path:
                subpaths.append(path2str(spath))
            print_paths.append(' + '.join(subpaths))
        else:
            print_paths.append(path2str(path))
    return sorted(print_paths, key=len, reverse=True)


def greedy_matching(print_pred_paths, print_gold_paths):
    hits = []
    for gold_path in print_gold_paths:
        if gold_path in print_pred_paths:
            hits.append(gold_path)
            print_pred_paths.remove(gold_path)
    return hits, print_pred_paths


def get_path_kb_ids(amr):

    # get children by parent index, removing re-entrancies
    # reentrancy_edges, _ = get_reentrancy_edges(amr)

    # Find all paths
    unknown_ids = []
    entity_ids = []
    kb_paths = []

    # Find NERs
    entity_ids = []
    for (src, label, tgt) in amr.edges:
        if amr.nodes[tgt] == 'name' and label == ':name':
            entity_ids.append(src)
    # Ignore NERs inside of NERs:
    new_entity_ids = []
    for ner_id in entity_ids:
        other = [e for e in entity_ids if e != ner_id]
        ancestor = get_ancestor_path(amr, ner_id, other)
        if any(a[-1] not in entity_ids for a in ancestor):
            new_entity_ids.append(ner_id)
    entity_ids = new_entity_ids

    # Loop over unknown nodes
    for uid, nname in amr.nodes.items():
        if nname != 'amr-unknown':
            continue
        unknown_ids.append(uid)

        # Loop over NER nodes
        for src in entity_ids:
            # Loop over paths joining unknown and NER
            for path in get_ancestor_path(amr, src, [uid]):
                if path[-1] == amr.root:
                    # if we did not find the node but just root, these two
                    # nodes are only connected through the root
                    for path2 in get_ancestor_path(amr, uid, []):
                        kb_paths.append(tuple([path[::-1], path2[::-1]]))
                else:
                    kb_paths.append(path[::-1])

    return unknown_ids, entity_ids, kb_paths


def get_ancestor_path(amr, node_id, ancestor_ids):
    '''
    Returns list of node ids coresponding to the path from ancestor_id to
    node_id (but built bottom up)
    '''

    # candidate paths x nodes in path
    # candidate_paths[-1][-1] is head of last path
    candidate_paths = [[node_id]]
    final_paths = []
    count = 0
    while count < 1000:

        # Remove all completed candidates (reached target node or root)
        while (
            candidate_paths
            and candidate_paths[-1][-1] in ancestor_ids + [amr.root]
        ):
            # add solved path to final list, exit if all candidates completed
            final_paths.append(candidate_paths.pop())

        # If no more candidates exit
        if candidate_paths == []:
            break

        # keep upwards, collect bifurcations for each new parent of the last
        # path
        new_candidates = []
        for parent in amr.parents(candidate_paths[-1][-1], edges=False):
            if parent in candidate_paths[-1]:
                # Skip re-entrancies in same
                continue
            new_candidates.append(candidate_paths[-1] + [parent])
        candidate_paths.pop()
        candidate_paths.extend(new_candidates)
        count += 1

    if count == 1000:
        set_trace(context=30)

    return final_paths


def main():

    # argument handling
    if len(sys.argv) != 3:
        print(f'\npython {sys.argv[0]} /path/to/pred_amr /path/to/gold_amr\n')
        exit(1)
    pred_amr_file, gold_amr_file = sys.argv[1:]

    # Read files
    pred_amrs = read_amr(pred_amr_file, ibm_format=True)
    gold_amrs = read_amr(gold_amr_file, ibm_format=True)
    assert len(pred_amrs) == len(gold_amrs)
    num_amrs = len(pred_amrs)

    # accumulate stats
    stats = []
    kb_stats = []
    for index in tqdm(range(num_amrs), desc='scoring'):

        pred_amr = pred_amrs[index]
        gold_amr = gold_amrs[index]

        # get the paths in forms of lists of ids
        print_gold_paths = get_print_paths(gold_amr, get_path_ids(gold_amr))
        print_pred_paths = get_print_paths(pred_amr, get_path_ids(pred_amr))

        # print them into paths with nodes and edges
        hits, misses = greedy_matching(print_pred_paths, print_gold_paths)

        # Tries, hits, misses
        stats.append([len(print_pred_paths), len(print_gold_paths), len(hits),
                      len(misses)])

        if 'amr-unknown' not in gold_amr.nodes.values():
            continue

        # KB path stats
        # print them into paths with nodes and edges
        # if index == 117:
        #    set_trace(context=30)
        print_gold_paths = get_print_paths(
            gold_amr,
            get_path_kb_ids(gold_amr)[2],
        )
        print_pred_paths = get_print_paths(
            pred_amr,
            get_path_kb_ids(pred_amr)[2],
        )

        hits, misses = greedy_matching(print_pred_paths, print_gold_paths)

        # Tries, hits, misses
        kb_stats.append([len(print_pred_paths), len(print_gold_paths),
                         len(hits), len(misses)])

    hits = sum([x[2] for x in stats])
    tries = sum([x[0] for x in stats])
    print(f'GGPA {hits * 100. /tries:.1f} %')

    hits = sum([x[2] for x in kb_stats])
    tries = sum([x[0] for x in kb_stats])
    print(f'GGPA-KB {hits * 100. /tries:.1f} %')
    set_trace(context=30)
    print()


if __name__ == '__main__':
    main()
