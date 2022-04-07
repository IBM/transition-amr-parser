import os
from collections import defaultdict, Counter

from transition_amr_parser.clbar import yellow_font, green_font
from transition_amr_parser.amr import normalize, add_alignments_to_penman
from ipdb import set_trace


class BadAlignModeSample(Exception):
    # Align mode is not 100% guaranteed to work. Use this Exception to capture
    # this case and reject the sample
    pass


def match_amrs(machine):

    gold2dec = machine.align_tracker.get_flat_map(reverse=True)
    dec2gold = {v[0]: k for k, v in gold2dec.items()}

    # sanity check: all nodes and edges there
    missing_nodes = [
        n for n in machine.gold_amr.nodes if n not in gold2dec
    ]

    # sanity check: all nodes and edges match
    edges = [
        (dec2gold[e[0]], e[1], dec2gold[e[2]])
        for e in machine.edges
        if e[0] in dec2gold and e[2] in dec2gold
    ]
    missing = set(machine.gold_amr.edges) - set(edges)
    excess = set(edges) - set(machine.gold_amr.edges)

    return missing_nodes, missing, excess


def check_gold_alignment(machine, trace=False, reject_samples=False):

    # sanity check
    gold2dec = machine.align_tracker.get_flat_map(reverse=True)
    dec2gold = {v[0]: k for k, v in gold2dec.items()}

    # sanity check: all nodes and edges there
    missing_nodes = [
        n for n in machine.gold_amr.nodes if n not in gold2dec
    ]
    if missing_nodes:
        if trace:
            print(machine)
            set_trace(context=30)
        elif reject_samples:
            raise BadAlignModeSample('Missing Nodes')

    # sanity check: all nodes and edges match
    edges = [
        (dec2gold[e[0]], e[1], dec2gold[e[2]])
        for e in machine.edges if (e[0] in dec2gold and e[2] in dec2gold)
    ]
    missing = set(machine.gold_amr.edges) - set(edges)
    excess = set(edges) - set(machine.gold_amr.edges)
    if bool(missing):
        if trace:
            print(machine)
            set_trace(context=30)
        else:
            if trace:
                print(machine)
                set_trace(context=30)
            elif reject_samples:
                raise BadAlignModeSample('Missing Edges')
    elif bool(excess):
        if trace:
            print(machine)
            set_trace(context=30)
        elif reject_samples:
            raise BadAlignModeSample('Excess Edges')


def print_and_break(machine, context=1):

    aligner = machine.align_tracker

    # SHIFT work-09
    dec2gold = aligner.get_flat_map()
    node_map = {
        k: green_font(f'{k}-{dec2gold[k][0]}')
        if k in dec2gold else yellow_font(k)
        for k in machine.nodes
    }
    # print
    os.system('clear')
    print()
    print(machine.state_str(node_map))
    print(aligner)
    set_trace(context=context)


def debug_align_mode(machine):

    gold2dec = machine.align_tracker.get_flat_map(reverse=True)
    dec2gold = {v[0]: k for k, v in gold2dec.items()}

    # sanity check: all nodes and edges there
    missing_nodes = [n for n in machine.gold_amr.nodes if n not in gold2dec]
    if missing_nodes:
        print_and_break(machine)

    # sanity check: all nodes and edges match
    edges = [(dec2gold[e[0]], e[1], dec2gold[e[2]]) for e in machine.edges]
    missing = sorted(set(machine.gold_amr.edges) - set(edges))
    excess = sorted(set(edges) - set(machine.gold_amr.edges))
    if bool(missing):
        print_and_break(machine)
    elif bool(excess):
        print_and_break(machine)


def get_ids_by_key(gold_nodes, gold_edges, gnids, forbid_nodes, ids,
                   twin_nodes):

    # loop over gold node is for same nname
    # source gold id given edge key
    ids_by_key = defaultdict(list)
    # target gold id given edge key
    hop_ids = defaultdict(list)
    # source node id we jumped to get to given target gnid
    new_backtrack = defaultdict(list)
    for gnid in gnids:
        for (gs, gl, gt) in gold_edges:

            # if we are hoping through a graph, avoid revisiting nodes
            if forbid_nodes:
                if gs == gnid and gt in forbid_nodes:
                    continue
                elif gt == gnid and gs in forbid_nodes:
                    continue
            # skip it does not involve this edge
            elif gnid not in [gs, gt]:
                continue

            # construct key from edge of current node
            # ignore re-entrant nodes
            if gs == gnid:  # and ra_count[gt] < 2:
                # child of nid
                if ids:
                    key = f'> {gl} {gt}'
                else:
                    key = f'> {gl} {normalize(gold_nodes[gt])}'
                if gt not in hop_ids[key]:
                    hop_ids[key].append(gt)
                new_backtrack[gt].append(gnid)
            elif gt == gnid:  # and la_count[gs] < 2:
                # parent of nid
                if ids:
                    key = f'{gs} {gl} <'
                else:
                    key = f'{normalize(gold_nodes[gs])} {gl} <'
                if gs not in hop_ids[key]:
                    hop_ids[key].append(gs)
                new_backtrack[gs].append(gnid)
            else:
                continue

            if gnid not in ids_by_key[key]:
                ids_by_key[key].append(gnid)

    # create key value pairs from edges by considering edges that identify one
    # or more nodes from the total of gnids
    edge_values = defaultdict(list)
    non_identifying_keys = []
    delete_backtracks = set()
    for key, key_gnids in ids_by_key.items():

        targets = hop_ids[key]

        if any(len(new_backtrack[t]) == 0 for t in targets):
            set_trace(context=30)
            pass

        if len(key_gnids) == 1:
            # uniquely identifies one gold_id
            edge_values[key] = list(key_gnids)[0:1]
            # there is no jump, so no backtrack
            for t in targets:
                delete_backtracks.add(t)
        else:

            if len(key_gnids) < len(gnids):
                # at least narrows down from the total set of possible
                # candidates
                edge_values[key] = sorted(key_gnids)

            elif (
                len(key_gnids) == len(gnids)
                and twin_nodes is not None
                and tuple(gnids) in twin_nodes.values()
            ):
                # it does not narrow, but it is the corner case of two
                # identical children.
                edge_values[key] = sorted(key_gnids)

            elif len(key_gnids) > len(gnids):
                # error
                set_trace(context=30)
                print()

            # if some target backtracks to more than one source, this is a
            # reentrancy and we ignore it for propagation
            for t in targets:
                if len(new_backtrack[t]) > 1:
                    delete_backtracks.add(t)
            # if not ignore:
            if delete_backtracks != set(targets):  # no target left
                non_identifying_keys.append(key)
            else:
                # set_trace(context=30)
                pass

    # delete backtracks that will not be used because edge alread identifies
    # uniquely or edge is a re-entrancy shared across nodes to disambiguate
    # also make all other backtracks single items
    new_backtrack = {
        k: v[0]
        for k, v in new_backtrack.items()
        if k not in delete_backtracks
    }

    return edge_values, non_identifying_keys, hop_ids, new_backtrack


def generate_matching_gold_hashes(gold_nodes, gold_edges, gnids, max_size=4,
                                  ids=False, forbid_nodes=None,
                                  backtrack=None, twin_nodes=None):
    """
    given graph defined by {gold_nodes} and {gold_edges} and node ids
    coresponding to nodes with same label {gnids} return structure that can be
    queried with a subgraph of the graph to identify each node in {gnids}
    """

    # if gnids == ['c3', 'c4']: set_trace(context=30)

    # get unique and non unique edges and from/to where we jump in the latter
    edge_values, non_identifying_keys, hop_ids, new_backtrack = \
        get_ids_by_key(
            gold_nodes, gold_edges, gnids, forbid_nodes, ids, twin_nodes
        )

    # if there are pending ids to be identified expand neighbourhood one
    # hop, indicate not to revisit nodes
    if (
        max_size > 1
        and len(non_identifying_keys)
        # if we are using gold node ids and not labels, size = 1 is all we need
        and not ids
    ):

        if backtrack is not None:
            new_backtrack = {
                k: backtrack[v]
                for k, v in new_backtrack.items()
                # TODO: Ensure this is missing due to allowing some
                # re-entrancies to propagat in get_ids_by_key
                if v in backtrack
            }

        for key in non_identifying_keys:

            hop_edge_values = generate_matching_gold_hashes(
                gold_nodes, gold_edges, sorted(hop_ids[key]), ids=ids,
                max_size=max_size - 1, forbid_nodes=gnids,
                backtrack=new_backtrack
            )
            # reached recursion depth and returns
            if hop_edge_values != {}:
                edge_values[key].append(hop_edge_values)

    if backtrack is not None:
        new_edge_values = defaultdict(list)
        for k, v in edge_values.items():
            for vi in v:
                if isinstance(vi, dict):
                    new_edge_values[k].append(vi)
                elif (
                    # re-entrant nodes have been removed from backtrack
                    vi in backtrack
                    and backtrack[vi] not in new_edge_values[k]
                ):
                    new_edge_values[k].append(backtrack[vi])
        edge_values = dict(new_edge_values)
    else:
        edge_values = dict(edge_values)

    return edge_values


def get_edge_keys(nodes, edges, nid, id_map=None):
    # get edges for this node
    key_nids = []
    for (s, l, t) in edges:

        # construct key from edge of current node
        if s == nid:
            # child of nid
            if id_map and t in id_map:
                key_nid = (f'> {l} {id_map[t][0]}', t)
            else:
                key_nid = (f'> {l} {normalize(nodes[t])}', t)
        elif t == nid:
            # parent of nid
            if id_map and s in id_map:
                key_nid = (f'{id_map[s][0]} {l} <', s)
            else:
                key_nid = (f'{normalize(nodes[s])} {l} <', s)
        else:
            continue

        key_nids.append(key_nid)

    return key_nids


def merge_candidates(new_candidates, candidates, reject_samples):

    if new_candidates == []:
        # not enough information to discrimnate further e.g. edges
        # not yet decoded
        return candidates

    elif (
        candidates == []
        or len(new_candidates) < len(candidates)
    ):
        # more restrictive disambiguation, take it
        # set_trace(context=30)
        return new_candidates

    elif len(new_candidates) == len(candidates):
        # equaly restrictive, intersect
        merge_cand = sorted(set(new_candidates) & set(candidates))
        if not merge_cand and reject_samples:
            raise BadAlignModeSample('Conflicting constrains for size=1')
        else:
            # keep the old
            return candidates
        # assert merge_cand, "Algorihm implementation error: There"\
        #    " can not be conflicting disambiguations"
        return merge_cand

    else:
        # less restrictive (redundant) ignore
        return candidates


def get_matching_gold_ids(nodes, edges, nid, edge_values, id_map=None,
                          twin_nodes=None, reject_samples=False):
    #
    # returns gold ids than can be uniquely assigned to node nid:
    # examples [], ['a1'], ['a1', 'a3']
    #
    # Not ambiguous
    #
    # edge_values = {None: ['a1']}
    #
    # Can be disambiguated with neighborhood 1
    #
    # edge_values = {('> :mod some'): ['a1']}
    #
    # Can be disambiguated with neighbouthood 2
    #
    # edge_values = {('paint-01 :ARG0 <'): {('> :mod some'): ['a1']}}
    #
    # Can partially be disambiguated with neighborhood 1 (assume > 2 gold_ids)
    #
    # edge_values = {('> :mod some'): ['a1', 'a3']}

    # edge_value: defaultdict(list) per nname

    if None in edge_values:
        return edge_values[None]

    key_nids = get_edge_keys(nodes, edges, nid, id_map=id_map)

    # match against pre-computed edge keys
    # collect all shallow disambiguations
    candidates = []
    for (key, knid) in key_nids:
        # get a match if any
        new_candidates = []
        for edge_value in edge_values.get(key, []):
            if not isinstance(edge_value, dict):
                new_candidates.append(edge_value)

        # apply merging criteria
        candidates = merge_candidates(
            new_candidates, candidates, reject_samples
        )

    # if we already found an exact match, no need to progress.
    # also, for node id mode, we only need size 1 neighborhood
    if id_map or len(candidates) == 1:
        return candidates

    # collect all recursive disambiguations. Note that they can only narrow
    # down our search or leave it as is, never widen it
    tree_keys = []
    for (key, knid) in key_nids:

        if key in tree_keys:
            continue

        # get a match if any
        for edge_value in edge_values.get(key, []):

            if isinstance(edge_value, dict):
                # if we hit neighbouthood > 1, process that one
                new_candidates = get_matching_gold_ids(
                    nodes, edges, knid, edge_value,
                    reject_samples=reject_samples
                )

                # apply merging criteria
                candidates = merge_candidates(
                    new_candidates, candidates, reject_samples
                )

                tree_keys.append(key)

    return candidates


def get_gold_node_hashes(gold_nodes, gold_edges, ids=False,
                         max_neigbourhood_size=6, twin_nodes=None):

    # cluster node ids by node label
    gnode_by_label = defaultdict(list)
    for gnid, gnname in gold_nodes.items():
        gnode_by_label[normalize(gnname)].append(gnid)

    # loop over clusters of node label and possible node ids
    rules = dict()
    for gnname, gnids in gnode_by_label.items():
        if len(gnids) == 1:
            # simple case: node label appears only one in graph
            rules[gnname] = [gnids[0]]
        else:

            # if normalize(gnname) == 'many': set_trace(context=30)

            rules[gnname] = generate_matching_gold_hashes(
                gold_nodes, gold_edges, gnids, max_size=max_neigbourhood_size,
                ids=ids, twin_nodes=twin_nodes
            )

            # sanity check, no empty rules
            # if rules[gnname] == {}:
            #    set_trace(context=30)
            #    pass

    return rules


class AlignModeTracker():
    '''
    Tracks alignment of decoded AMR in align-mode to the corresponding gold AMR
    '''

    def __init__(self, gold_amr, reject_samples=False):

        self.gold_amr = gold_amr

        # If set raise BadAlignModeSample if a conflict is found
        self.reject_samples = reject_samples

        # assign gold node ids to decoded node ids based on node label
        gnode_by_label = defaultdict(list)
        for gnid, gnname in gold_amr.nodes.items():
            gnode_by_label[normalize(gnname)].append(gnid)

        # collect ambiguous and certain mappings separately in the structure
        # {node_label: [[gold_ids], [decoded_ids]]}
        # self.ambiguous_gold_id_map = defaultdict(lambda: [[], []])
        self.gold_id_map = dict()
        for gnname, gnids in gnode_by_label.items():
            # examples:
            # {
            #    12: ['h'],
            #    3: ['h2', 'h3'],
            #    7: ['h2', 'h3'],
            # None: ['h4']
            # }
            self.gold_id_map[gnname] = {None: gnids}

        # re-entrancy counts for both gold and decoded
        # {node_id: number_of_parents}
        self.num_gold_parents = {
            n: len(self.gold_amr.parents(n)) for n in self.gold_amr.nodes
        }
        self.num_dec_parents = {}

        # there can be more than one edge between two nodes e.g. John hurt
        # himself
        # there are also rare cases of two identical children
        # self.repeated_edge_names
        self.twin_nodes = defaultdict(list)
        for gs, gl, gt in gold_amr.edges:
            self.twin_nodes[(gs, gl, gold_amr.nodes[gt])].append(gt)
        for k in [k for k, nids in self.twin_nodes.items() if len(nids) > 1]:
            self.twin_nodes[k] = tuple(sorted(self.twin_nodes[k]))
        for k in [k for k, nids in self.twin_nodes.items() if len(nids) == 1]:
            del self.twin_nodes[k]

        # a data structure list(dict|list) that can disambiguate a decoded node
        # (assign it to it gold node, give sufficient decoded edges). This will
        # search entire graph neighbourhood
        self.dec_neighbours = get_gold_node_hashes(
            gold_amr.nodes, gold_amr.edges, max_neigbourhood_size=6,
            twin_nodes=self.twin_nodes
        )
        # the same but given gold_ids already alignet to decoded ids. this only
        # needs size 1
        self.gold_neighbours = get_gold_node_hashes(
            gold_amr.nodes, gold_amr.edges, ids=True,
            twin_nodes=self.twin_nodes
        )

        # sanity check all nodes can be disambiguated with current conditions
        for nname in self.dec_neighbours:
            if (
                self.dec_neighbours[nname] == []
                and self.gold_neighbours[nname] == []
            ):
                set_trace(context=30)
                print()

        # there can be more than one edge between two nodes e.g. John hurt
        # himself
        # there are also rare cases of two identical children
        self.num_edges_by_node_pair = Counter()
        self.num_edges_by_node_pair.update(
            (gs, gt) for gs, _, gt in gold_amr.edges
        )

        # this will hold decoded edges aligned to gold edges, including
        # ambiguous cases (many aligned ot many)
        self.decoded_to_goldedges = {}
        # this will hold which postential decoded edges would correspond to
        # each pending gold edge
        self.goldedges_to_candidates = {}

        self._edges_by_child = defaultdict(list)
        self._edges_by_parent = defaultdict(list)

    def __str__(self):

        string = ''
        for nname, pairs in self.gold_id_map.items():

            # exclude pairs that will not be ambiguous
            if len(pairs) == 1 and len(pairs.values()) == 1:
                continue

            string += f'"{nname}" '
            for nid, gnids in pairs.items():
                if nid is None:
                    # unassigned gold node ids
                    str2 = ' '.join(gnids)
                    string += '['
                    string += yellow_font(f'{str2}')
                    string += '] '

                elif len(gnids) > 1:
                    # assigned but not 1-1
                    str2 = ' '.join(gnids)
                    string += f'{nid}: '
                    string += '['
                    string += yellow_font(f'{str2}')
                    string += '] '

                else:
                    # 1-1 assigned
                    str2 = ' '.join(gnids)
                    string += green_font(f'{nid}-{str2} ')

        string += ' \n\n'
        return string

    def get_neighbour_disambiguation(self, amr):

        # cluster node ids by node label
        gnode_by_label = defaultdict(list)
        for gnid, gnname in amr.nodes.items():
            gnode_by_label[normalize(gnname)].append(gnid)

        # get all neighbours of each node
        id_neighbours = defaultdict(list)
        for nid, nname in amr.nodes.items():
            if len(gnode_by_label[nname]) > 1:
                id_neighbours[nid] = []
                for (s, l, t) in amr.edges:
                    if s == nid or t == nid:
                        id_neighbours[nid].append(
                            (amr.nodes[s], l, amr.nodes[t])
                        )

        # get the unique identifying edges, i.e. the edegs that only appear as
        # relatives of one of the nodes in a set that shares same node label
        for nname, nids in gnode_by_label.items():
            if len(nids) == 1:
                continue
            edge_counts = Counter([e for n in nids for e in id_neighbours[n]])
            for nid in nids:
                id_neighbours[nid] = [
                    e for e in id_neighbours[nid] if edge_counts[e] == 1
                ]

        return id_neighbours

    def get_flat_map(self, reverse=False, ambiguous=False, no_list=False):
        '''
        Get a flat map from gold_id_map, optionally including ambiguous pairs
        '''

        # Normal map is ordered and maps one to one
        dec2gold = defaultdict(list)
        for nname, pairs in self.gold_id_map.items():
            for nid, gnids in pairs.items():
                if (len(gnids) == 1 or ambiguous) and nid is not None:
                    if reverse:
                        for gnid in gnids:
                            dec2gold[gnid].append(nid)
                    else:
                        dec2gold[nid] = gnids

        # cast into single item, if possible
        if no_list:
            assert max(map(len, dec2gold.values())) == 1
            dec2gold = {k: v[0] for k, v in dec2gold.items()}

        return dec2gold

    def _map_decoded_and_gold_ids_by_propagation(self, machine):
        '''
        Given partial aligned graph and gold graph map each prediced node id to
        a gold id by matching node labels and edges to neighbours
        '''

        # get gold to decoded node map without node names
        dec2gold = self.get_flat_map()

        # check if nodes can be disambiguated by propagating ids through edges
        for (s, l, t) in machine.edges:
            if s in dec2gold and t not in dec2gold:

                # We can have more than one edge meeting
                # conditions here, in this case, we skip
                candidates = []
                nname = normalize(machine.nodes[t])
                for gt, gl in self.gold_amr.children(dec2gold[s]):
                    if (
                        normalize(self.gold_amr.nodes[gt]) == machine.nodes[t]
                        and gl == l
                        # FIXME: if decoded candidate has some subgraph, check
                        # it is consistent with the alignment that we are going
                        # to make
                        # <-- HERE
                    ):
                        candidates.append((gt, gl))

                if len(candidates) == 1:
                    gt, gl = candidates[0]
                    self.disambiguate_pair(nname, gt, t)
                    # also update decoded -> gold map and ignore other
                    # edges. FIXME: This is a greedy decision.
                    dec2gold[t] = gt

            elif t in dec2gold and s not in dec2gold:
                # Due to co-reference, we can have more than one edge meeting
                # conditions here, in this case, we skip
                candidates = []
                nname = normalize(machine.nodes[s])
                for gs, gl in self.gold_amr.parents(dec2gold[t]):
                    if (
                        normalize(self.gold_amr.nodes[gs]) == machine.nodes[s]
                        and gl == l
                    ):
                        candidates.append((gs, gl))

                if len(candidates) == 1:
                    gs, gl = candidates[0]
                    self.disambiguate_pair(nname, gs, s)
                    # also update decoded -> gold map and ignore other
                    # edges. FIXME: This is a greedy decision.
                    dec2gold[s] = gs

    def _map_decoded_and_gold_ids_by_context(self, machine):

        # Here we differentiate node ids, unique identifiers of nodes inside a
        # graph, from labels, node names, which may be repeatable.
        # when a node is predicted, it may be ambiguous to which gold node it
        # corresponds. Only if the label is unique or its graph position is
        # unique we can determine it. So we need to wait until enough edges are
        # available

        # disambiguate and store expansion list
        match_updates = []
        for gnname, pairs in self.gold_id_map.items():
            for nid, gnids in pairs.items():

                # not ambiguous or no new nodes to disambiguate
                if len(gnids) == 1 or nid is None:
                    continue

                # if len(machine.action_history) > 18 and gnname == 'name':
                #    print_and_break(machine, 1)

                # disambiguate by neighbourhood of decoded nodes
                matches = get_matching_gold_ids(
                    machine.nodes, machine.edges, nid,
                    self.dec_neighbours[gnname],
                    reject_samples=self.reject_samples
                )

                # for twin nodes any selection will be ok
                if tuple(sorted(matches)) in self.twin_nodes.values():
                    if None in pairs:
                        matches = pairs[None][0:1]

                # disambiguate by neighbourhood of aligned gold nodes
                prop_matches = get_matching_gold_ids(
                    machine.nodes, machine.edges, nid,
                    self.gold_neighbours[gnname],
                    id_map=self.get_flat_map(),
                    reject_samples=self.reject_samples
                )

                # for twin nodes any selection will be ok
                if tuple(sorted(prop_matches)) in self.twin_nodes.values():
                    if matches:
                        prop_matches = matches
                    elif None in pairs:
                        prop_matches = pairs[None][0:1]

                # if len(machine.action_history) > 21:
                #    print_and_break(machine, 30)

                # keep more restrictive disambiguation
                if (
                    prop_matches
                    and (matches == [] or len(prop_matches) < len(matches))
                ):
                    matches = prop_matches

                if matches:
                    match_updates.append((gnname, nid, matches))

        # perform the updates
        for (gnname, nid, matches) in match_updates:
            # FIXME: tenporary solution to get stats
            # if bool(set(self.gold_id_map[gnname][nid]) & set(matches)):
            #    continue
            self.disambiguate_pair(gnname, nid, matches)

    def update(self, machine):
        '''
        Update state of aligner given latest state machine state
        '''

        # update neighbourhood auxiliary structures to descrbe decoded graph
        self._edges_by_child = defaultdict(list)
        for (source, edge_name, target) in machine.edges:
            self._edges_by_child[target].append((source, edge_name))
        self._edges_by_parent = defaultdict(list)
        for (source, edge_name, target) in machine.edges:
            self._edges_by_parent[source].append((target, edge_name))

        # note that actions predict node labels, not node ids so during partial
        # decoding we may not know which gold node corresponds to which decoded
        # node until a number of edges have been predicted.

        # map added nodes that are not ambiguous
        dec2gold = self.get_flat_map(ambiguous=True)
        for nid, nname in machine.nodes.items():
            if nname not in self.gold_id_map:
                # e.g. <unk>
                set_trace(context=30)
                raise Exception()

            # if this is a new decoded node add it to the map dict
            if nid not in self.gold_id_map[nname]:
                if (
                    len(self.dec_neighbours[nname]) == 1
                    and not isinstance(self.dec_neighbours[nname], dict)
                ):
                    # 1-1 mapping, assign directly
                    gnid = self.dec_neighbours[nname]
                    self.disambiguate_pair(nname, nid, gnid)
                else:
                    # otherwise add to unassigned nodes
                    self.disambiguate_pair(nname, nid, [])

        # if we found root, set it
        # This will affect the machine
        gold_to_dec_ids = self.get_flat_map(reverse=True)
        if machine.gold_amr.root in gold_to_dec_ids:
            machine.root = gold_to_dec_ids[machine.gold_amr.root][0]

        # update re-entrancy counts
        # TODO: move before previous block
        num_dec_parents = Counter()
        for (s, l, t) in machine.edges:
            if all(
                n != self.gold_amr.root
                and num_dec_parents[t] == self.num_gold_parents[n]
                for n in dec2gold[t]
            ) and self.reject_samples:
                # at least one option should have less number of parents
               raise BadAlignModeSample('Invalid re-entrancy counts')

            num_dec_parents[t] += 1
        self.num_dec_parents = num_dec_parents

        # update gold and decoded graph alignments
        # propagating from aligned pairs to unaligned ones
        # self._map_decoded_and_gold_ids_by_propagation(machine)
        # by context matching
        self._map_decoded_and_gold_ids_by_context(machine)

        # when only one ambiguous node is left, its not ambiguous any more
        # for nname in self.gold_id_map.keys():
        #    self.exclusion_disambiguation(nname)

        # update edge alignments
        # TODO: maybe use node alignment here for propagation (iteratively)
        self._map_decoded_and_gold_edges(machine)

    def disambiguate_pair(self, nname, decoded_id, gold_ids):

        # TODO: more checks
        assert isinstance(gold_ids, list)
        assert isinstance(decoded_id, str) or isinstance(decoded_id, int)

        # fast exit
        if decoded_id in self.gold_id_map[nname] and gold_ids != []:

            old_gnids = self.gold_id_map[nname][decoded_id]
            intersection = sorted(set(gold_ids) & set(old_gnids))

            if set(gold_ids) == set(old_gnids):
                # we already have assigned this
                return

            elif set(gold_ids) < set(old_gnids):
                # more restrictive assignment, we will take it
                pass

            elif bool(intersection):

                # only partially overlapping
                # keep the intersection of gold_ids
                gold_ids = [n for n in gold_ids if n in intersection]
                if set(gold_ids) == set(old_gnids):
                    # we already have assigned this
                    # FIXME: isnt this set(gold_ids) < set(old_gnids)?
                    # just ignore this call
                    return

            elif self.reject_samples:

                # update does conflict with previous. Normally the invalid move
                # has happened some actions previous, so here it is too late
                raise BadAlignModeSample('Conflicting constraints')

            else:

                # just ignore this call
                return

        # get lists of decoded ids and gold ids that are already 1-1 mapped
        assigned_nids = []
        assigned_gnids = []
        unassigned_gnids = []
        for nid, gnids in self.gold_id_map[nname].items():
            if len(gnids) == 1 and nid is not None:
                # 1-1 map
                if decoded_id == nid:
                    # already assigned
                    set_trace(context=30)
                assigned_nids.append(nid)
                if gnids[0] not in assigned_gnids:
                    assigned_gnids.append(gnids[0])
            else:
                for gnid in gnids:
                    if gnid not in unassigned_gnids:
                        unassigned_gnids.append(gnid)

        # if we are assigning something, remove gold_ids that have been
        # assigned
        if gold_ids:
            gold_ids = [n for n in gold_ids if n not in assigned_gnids]
            if not bool(gold_ids):
                if self.reject_samples:
                    raise BadAlignModeSample('gold_id already assigned')
                else:
                    # just ignore this call
                    return

        self.sanity_checks(
            nname, decoded_id, gold_ids, assigned_nids, assigned_gnids
        )

        # if this a new decoded node asign to it the gold nodes under None or
        # just the unassigned ids
        if gold_ids == []:
            if None in self.gold_id_map[nname]:
                self.gold_id_map[nname][decoded_id] = \
                    list(self.gold_id_map[nname][None])
            else:
                # with no further info assign it all non 1-1-mapped nodes
                self.gold_id_map[nname][decoded_id] = unassigned_gnids

        else:

            # assign one or more gold ids
            # e.g. {2: ['h1', 'h2']}
            self.gold_id_map[nname][decoded_id] = gold_ids

            # apply exlusion principle and remove gold ids from the None group
            # FIXME: propagate this information to existing neighbours
            self.cleanup_map(nname, gold_ids, assigned_nids)

    def sanity_checks(self, nname, decoded_id, gold_ids, assigned_nids,
                      assigned_gnids):

        if gold_ids == []:

            # Add a new decoded node to its bucket unassigned

            # check node not has been decoded yet
            if decoded_id in self.gold_id_map[nname]:
                set_trace(context=30)
                raise Exception(f'{decoded_id} already decoded')

            if None not in self.gold_id_map[nname]:
                set_trace(context=30)
                raise Exception(f'No more free nodes')

        else:

            # assign one or more gold ids to decoded_id

            # we have not assigned these gold ids before
            if set(gold_ids) & set(assigned_gnids):
                set_trace(context=30)
                raise Exception(f'{gold_ids} already disambiguated')

            if decoded_id in self.gold_id_map[nname]:

                old_gnids = self.gold_id_map[nname][decoded_id]

                # check we have not 1-1-mapped this already
                if decoded_id in assigned_nids:
                    set_trace(context=30)
                    raise Exception(f'{decoded_id} already disambiguated')

                # check this is not a less restrictive disambiguation
                if (
                    old_gnids and len(old_gnids) <= len(gold_ids)
                    and set(gold_ids) != set(old_gnids)
                ):
                    set_trace(context=30)
                    raise Exception(f'Less restrictive disambiguation')

    def cleanup_map(self, nname, gold_ids, assigned_gnids):

        # if there is a 1-1 assignment remove that node from the None list
        if (
            None in self.gold_id_map[nname]
            and len(gold_ids) == 1
            and gold_ids[0] in self.gold_id_map[nname][None]
        ):
            self.gold_id_map[nname][None].remove(gold_ids[0])
            if self.gold_id_map[nname][None] == []:
                del self.gold_id_map[nname][None]

        # if nname == 'country':
        #    set_trace(context=30)

        # if it is a 1-1 assignment remove from all other assignments. This
        # needs to be done repeteadly as we may generate new unique maps
        while assigned_gnids:
            mapped_gnid = assigned_gnids.pop()
            for nid2, gnids in self.gold_id_map[nname].items():
                if len(gnids) > 1 and mapped_gnid in gnids:
                    # remove this one2one mapped node from all other maps
                    # this other maps include None, which may have only that
                    # gold id (len == 1)
                    self.gold_id_map[nname][nid2].remove(mapped_gnid)
                    if len(self.gold_id_map[nname][nid2]) == 1:
                        # we assigned another node, we need to continue
                        assigned_gnids.append(
                            self.gold_id_map[nname][nid2][0]
                        )

                    # TODO: if nid2 has ambiguous neighbours, they may be less
                    # ambiguous

        # remove None if empty
        if (
            None in self.gold_id_map[nname]
            and self.gold_id_map[nname][None] == []
        ):
            del self.gold_id_map[nname][None]

    def get_potential_gold_edges(self, machine, gold_to_dec_ids):

        potential_gold_edges = []
        for (gold_s_id, gold_e_label, gold_t_id) in self.gold_amr.edges:

            if (
                gold_s_id not in gold_to_dec_ids
                or gold_t_id not in gold_to_dec_ids
            ):
                # no decoded candidates for this edge yet
                continue

            if not (
                # one or more possible RAs
                any(
                    n in gold_to_dec_ids[gold_s_id]
                    for n in machine.node_stack[:-1]
                ) and machine.node_stack[-1] in gold_to_dec_ids[gold_t_id]
                # one or more possible LAs
                or any(
                    n in gold_to_dec_ids[gold_t_id]
                    for n in machine.node_stack[:-1]
                ) and machine.node_stack[-1] in gold_to_dec_ids[gold_s_id]
            ):
                # non of the ids is at the top of the stack
                continue

            potential_gold_edges.append((gold_s_id, gold_e_label, gold_t_id))

        return potential_gold_edges

    def _map_decoded_and_gold_edges(self, machine):

        # get a map from every gold node to every potential aligned decoded
        gold_to_dec_ids = self.get_flat_map(reverse=True, ambiguous=True)
        # get potential gold edges to predicty given stack content
        potential_gold_edges = self.get_potential_gold_edges(
            machine, gold_to_dec_ids
        )

        # note that during partial decoding we may have multiple possible
        # decoded edges for each gold edge (due to ambiguous node mapping
        # above) Here we cluster gold edges by same potential decoded edges.
        self.decoded_to_goldedges = defaultdict(list)
        self.goldedges_to_candidates = defaultdict(list)
        for (gold_s_id, gold_e_label, gold_t_id) in potential_gold_edges:

            # store decoded <-> gold cluster
            key = []
            used_nids = []
            for nid in gold_to_dec_ids[gold_s_id]:
                for nid2 in gold_to_dec_ids[gold_t_id]:
                    if (nid, gold_e_label, nid2) in machine.edges:
                        # this possible disambiguation is already decoded
                        key.append((nid, gold_e_label, nid2))
                        used_nids.append(nid)
                        # targets may be used more than once due to
                        # re-entrancies
            self.decoded_to_goldedges[tuple(key)].append(
                (gold_s_id, gold_e_label, gold_t_id)
            )

            # if len(machine.action_history) > 7:
            #    print(machine)
            #    set_trace(context=30)

            # store potential decodable edges for this gold edge. Note that we
            # can further disambiguate for this given gold edge
            for nid in gold_to_dec_ids[gold_s_id]:

                if nid in used_nids:
                    # if we used this already above, is not a possible decoding
                    continue
                for nid2 in gold_to_dec_ids[gold_t_id]:
                    if (
                        # we need to take into account re-entrancies
                        self.num_dec_parents[nid2] == \
                            self.num_gold_parents[gold_t_id]
                        or (nid, gold_e_label, nid2) in machine.edges
                        or nid == nid2
                    ):
                        # if we used this already above, is not a possible
                        # decoding
                        continue

                    # constrain to gold number of edges per node pairs.
                    if (
                        self.num_edges_by_node_pair[(gold_s_id, gold_t_id)] ==
                        sum((e[0], e[2]) == (nid, nid2) for e in machine.edges)
                    ):
                        continue

                    # this is a potential decoding option
                    self.goldedges_to_candidates[
                        (gold_s_id, gold_e_label, gold_t_id)
                    ].append((nid, gold_e_label, nid2))

    def get_missing_nnames(self, repeat=False):

        # Add from ambiguous and normal mappings the nodes that have not yet
        # been decoded
        missing_gold_nodes = []
        for nname, pairs in self.gold_id_map.items():

            # get the total number of gold nodes with this node and the number
            # of decoded nodes
            total_gnids = len(set([y for x in pairs.values() for y in x]))
            total_decoded = len([k for k in pairs.keys() if k is not None])
            num_missing = total_gnids - total_decoded
            assert num_missing >= 0
            if num_missing:
                if repeat:
                    missing_gold_nodes.extend([nname] * num_missing)
                else:
                    missing_gold_nodes.append(nname)

        return missing_gold_nodes

    def get_missing_edges(self, machine):

        # if there are N gold edges and N possible decoded edges (even if we do
        # not know which is which) consider those gold edges complete,
        # otherwise add to candidates
        missing_gold_edges = []
        for dec_edges, gold_edges in self.decoded_to_goldedges.items():
            if len(dec_edges) != len(gold_edges):
                missing_gold_edges.extend(gold_edges)
            elif len(dec_edges) > len(gold_edges):
                set_trace(context=30)
                raise Exception()

        # get chilldren names and node count for this graph
        # TODO: we could reuse the ones from neighborhood
        edge_count = Counter()
        child_names = defaultdict(list)
        parent_names = defaultdict(list)
        for (s, l, t) in machine.edges:
            child_names[s].append((l, normalize(machine.nodes[t])))
            parent_names[t].append((l, normalize(machine.nodes[s])))
            edge_count.update([(s, t)])

        # expand to all possible disambiguations
        expanded_missing_gold_edges = []
        for gold_edge in missing_gold_edges:
            for decoded_edge in self.goldedges_to_candidates[gold_edge]:

                dec_s, dec_l, dec_t = decoded_edge

                # avoid having more number of edges between two nodes than in
                # gold
                # pair_key = (dec_t, dec_s) if dec_l.endswith('-of') \
                #    else (dec_s, dec_t)
                if (
                    edge_count[(dec_s, dec_t)] ==
                    self.num_edges_by_node_pair[(gold_edge[0], gold_edge[2])]
                ):
                    continue

                nname_t = normalize(machine.nodes[dec_t])
                nname_s = normalize(machine.nodes[dec_s])
                child_name = (dec_l, nname_t)
                full_child_name = (gold_edge[0], dec_l, nname_t)
                parent_name = (dec_l, nname_s)
                full_parent_name = (nname_s, dec_l, gold_edge[2])

                num_child = sum([c == child_name for c in child_names[dec_s]])
                num_parent = sum([c == parent_name for c in parent_names[dec_t]])

                # avoid existing edges, in some odd cases same edge may appear
                # multiple times
                if (
                (
                    num_child > 0 and
                    num_child
                    == len(self.twin_nodes.get(full_child_name, [None]))
                ) and (
                    num_parent > 0 and
                    num_parent
                    == len(self.twin_nodes.get(full_parent_name, [None]))
                )):
                    # this edge already exists both as a child and a parent
                    continue

                expanded_missing_gold_edges.append((dec_s, dec_l, dec_t))

        return expanded_missing_gold_edges

    def add_alignments_to_penman(self, machine):

        gold2dec = self.get_flat_map()

        # This info is checked at close, so no need to raise here
        # if (
        #    any(nid not in gold2dec for nid in machine.alignments)
        #    and self.reject_samples
        # ):
        #    # set_trace(context=30)
        #    raise BadAlignModeSample('Missing node')

        alignments = {
            gold2dec[nid][0]: pos
            for nid, pos in machine.alignments.items()
            if nid in gold2dec
        }

        return add_alignments_to_penman(
            self.gold_amr.penman,
            alignments,
            string=True
        )
