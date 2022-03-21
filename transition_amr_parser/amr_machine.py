import json
import argparse
import os
from functools import partial
import re
from copy import deepcopy
from itertools import chain
from collections import defaultdict, Counter

from tqdm import tqdm
import numpy as np
from transition_amr_parser.io import (
    AMR,
    read_amr,
    read_tokenized_sentences,
    write_tokenized_sentences,
)
from transition_amr_parser.amr import add_alignments_to_penman

from transition_amr_parser.clbar import yellow_font, green_font, clbar
from ipdb import set_trace

# change the format of pointer string from LA(label;pos) -> LA(pos,label)
la_regex = re.compile(r'>LA\((.*),(.*)\)')
ra_regex = re.compile(r'>RA\((.*),(.*)\)')
arc_regex = re.compile(r'>[RL]A\((.*),(.*)\)')
la_nopointer_regex = re.compile(r'>LA\((.*)\)')
ra_nopointer_regex = re.compile(r'>RA\((.*)\)')
arc_nopointer_regex = re.compile(r'>[RL]A\((.*)\)')


def debug_align_mode(machine):

    gold2dec = machine.align_tracker.get_flat_map(reverse=True)
    dec2gold = {v: k for k, v in gold2dec.items()}

    # sanity check: all nodes and edges there
    missing_nodes = [n for n in machine.gold_amr.nodes if n not in gold2dec]
    if missing_nodes:
        set_trace(context=30)
        return

    # sanity check: all nodes and edges match
    edges = [(dec2gold[e[0]], e[1], dec2gold[e[2]]) for e in machine.edges]
    missing = set(machine.gold_amr.edges) - set(edges)
    excess = set(edges) - set(machine.gold_amr.edges)
    if bool(missing):
        set_trace(context=30)
        print()
    elif bool(excess):
        set_trace(context=30)
        print()


def print_and_break(context, aligner, machine):

    # SHIFT work-09
    dec2gold = aligner.get_flat_map()
    node_map = {
        k: green_font(f'{k}-{dec2gold[k]}')
            if k in dec2gold else yellow_font(k)
        for k in machine.nodes
    }
    print(machine.state_str(node_map))
    print(aligner)
    set_trace(context=context)
    # set_trace()


def generate_matching_gold_hashes(gold_nodes, gold_edges, gnids, max_size=4,
                                  ids=False, forbid_nodes=None,
                                  backtrack=None):
    """
    given graph defined by {gold_nodes} and {gold_edges} and node ids
    coresponding to nodes with same label {gnids} return structure that can be
    queried with a subgraph of the graph to identify each node in {gnids}
    """

    # loop over gold node is for same nname
    ids_by_key = defaultdict(set)
    hop_ids = defaultdict(set)
    new_backtrack = {}
    for gnid in gnids:
        for (gs, gl, gt) in gold_edges:

            # if we are hoping through a graph, avoid revisiting nodes
            if forbid_nodes:
                if gs == gnid and gt in forbid_nodes:
                    continue
                elif gt == gnid and gs in forbid_nodes:
                    continue

            # construct key from edge of current node
            if gs == gnid:
                # child of nid
                if ids:
                    key = f'> {gl} {gt}'
                else:
                    key = f'> {gl} {normalize(gold_nodes[gt])}'
                hop_ids[key].add(gt)
                new_backtrack[gt] = gnid
            elif gt == gnid:
                # parent of nid
                if ids:
                    key = f'{gs} {gl} <'
                else:
                    key = f'{normalize(gold_nodes[gs])} {gl} <'
                hop_ids[key].add(gs)
                new_backtrack[gs] = gnid
            else:
                continue
            ids_by_key[key].add(gnid)

    # create key value pairs from edges by considering edges that identify nine
    # or more nodes from the total of gnids
    edge_values = defaultdict(list)
    found_gnids = set()
    non_identifying_keys = []
    for key, key_gnids in ids_by_key.items():

        if len(key_gnids) == 1:
            # uniquely identifies one gold_id
            edge_values[key] = list(key_gnids)[0:1]
            found_gnids |= key_gnids
        elif len(key_gnids) < len(gnids):
            # uniquely identifies two or more gold_ids
            edge_values[key] = list(key_gnids)
            non_identifying_keys.append(key)
        else:
            # does not indentify anything
            non_identifying_keys.append(key)

    # if there are pending ids to be identified expand neighbourhood one
    # hop, indicate not to revisit nodes
    if (
        len(found_gnids) < len(gnids)
        and max_size > 1
        and len(non_identifying_keys)
        # if we are using gold node ids and not labels, size = 1 is all we need
        and not ids
    ):

        if backtrack is not None:
            new_backtrack = {k: backtrack[v] for k, v in new_backtrack.items()}

        for key in non_identifying_keys:

            hop_edge_values = generate_matching_gold_hashes(
                gold_nodes, gold_edges, list(hop_ids[key]), ids=ids,
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
                elif backtrack[vi] not in new_edge_values[k]:
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
                key_nid = (f'> {l} {id_map[t]}', t)
            else:
                key_nid = (f'> {l} {normalize(nodes[t])}', t)
        elif t == nid:
            # parent of nid
            if id_map and s in id_map:
                key_nid = (f'{id_map[s]} {l} <', s)
            else:
                key_nid = (f'{normalize(nodes[s])} {l} <', s)
        else:
            continue

        key_nids.append(key_nid)

    return key_nids


def get_matching_gold_ids(nodes, edges, nid, edge_values, id_map=None):
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
        for edge_value in edge_values.get(key, []):
            if not isinstance(edge_value, dict):
                candidates.append(edge_value)

    # for node id mode, we only need size 1 neighborhood
    if id_map:
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
                new_candidates = \
                    get_matching_gold_ids(nodes, edges, knid, edge_value)
                if new_candidates == []:
                    # not enough information to discrimnate further e.g. edges
                    # not yet decoded
                    pass
                elif (
                    candidates == []
                    or len(new_candidates) < len(candidates)
                ):
                    # more restrictive disambiguation, take it
                    # set_trace(context=30)
                    candidates = new_candidates
                elif len(new_candidates) == len(candidates):
                    # equaly restrictive, intersect
                    # set_trace(context=30)
                    candidates = sorted(set(new_candidates) & set(candidates))
                    assert candidates, "Algorihm implementation error: There"\
                        " can not be conflicting disambiguations"
                else:
                    # less restrictive (redundant) ignore
                    pass

                tree_keys.append(key)

    return candidates


def get_gold_node_hashes(gold_nodes, gold_edges, ids=False):

    # cluster node ids by node label
    gnode_by_label = defaultdict(list)
    for gnid, gnname in gold_nodes.items():
        gnode_by_label[normalize(gnname)].append(gnid)

    # loop over clusters of node label and possible node ids
    max_neigbourhood_size = 3
    rules = dict()
    for gnname, gnids in gnode_by_label.items():
        if len(gnids) == 1:
            # simple case: node label appears only one in graph
            rules[gnname] = [gnids[0]]
        else:
            rules[gnname] = generate_matching_gold_hashes(
                gold_nodes, gold_edges, gnids, max_size=max_neigbourhood_size,
                ids=ids
            )

    return rules


def red_background(string):
    return "\033[101m%s\033[0m" % string


# def get_gold_node_hashes(gold_nodes, gold_edges):
#
#     # cluster node ids by node label
#     gnode_by_label = defaultdict(list)
#     for gnid, gnname in gold_nodes.items():
#         gnode_by_label[normalize(gnname)].append(gnid)
#
#     # get all neighbours of each node
#     id_neighbours = defaultdict(list)
#     nname_neighbours = defaultdict(list)
#     for nid, nname in gold_nodes.items():
#         id_neighbours[nid] = []
#         for (s, l, t) in gold_edges:
#             if s == nid:
#                 id_neighbours[nid].append(((s, l, t), t))
#                 nname_neighbours[nid].append((gold_nodes[s], l, gold_nodes[t]))
#             elif t == nid:
#                 id_neighbours[nid].append(((s, l, t), s))
#
#     # visit increasing neighbourhoods until you get a hash that can
#     node_hash = {}
#     max_neigbourhood_size = 3
#     for nname, nids in gnode_by_label.items():
#
#         if len(nids) == 1:
#             # single node, trivial
#             continue
#
#         pending_nids = nids
#
#         neighbourhood = dict(id_neighbours)
#         for size in range(max_neigbourhood_size):
#
#             for s in range(size - 1):
#                 for nid in pending_nids:
#                     neighbourhood[nid]
#
#             edge_counts = Counter([e for pending_nids in nids for e in id_neighbours[n]])
#
#             for nid in pending_nids:
#                 # any edge appearing in only one node is a valid hash
#                 # FIXME: also same edge 2+ for the same node
#                 hash_edges = [
#                     e for e in id_neighbours[nid] if edge_counts[e] == 1
#                 ]
#                 if hash_edges:
#
#                     node_hash[nid] = hash_edges
#                     pending_nids.remove(nid)
#
#     return id_neighbours


def graph_alignments(unaligned_nodes, amr):
    """
    Shallow alignment fixer: Inherit the alignment of the last child or first
    parent. If none of these is aligned the node is left unaligned
    """

    fix_alignments = {}
    for (src, _, tgt) in amr.edges:
        if (
            src in unaligned_nodes
            and amr.alignments[tgt] is not None
            and max(amr.alignments[tgt])
                > fix_alignments.get(src, 0)
        ):
            # # debug: to justify to change 0 to -1e6 for a test data corner
            # case; see if there are other cases affected
            # if max(amr.alignments[tgt]) <= fix_alignments.get(src, 0):
            #     breakpoint()
            fix_alignments[src] = max(amr.alignments[tgt])
        elif (
            tgt in unaligned_nodes
            and amr.alignments[src] is not None
            and min(amr.alignments[src])
                < fix_alignments.get(tgt, 1e6)
        ):
            fix_alignments[tgt] = max(amr.alignments[src])

    return fix_alignments


def fix_alignments(gold_amr):

    # Fix unaligned nodes by graph vicinity
    unaligned_nodes = set(gold_amr.nodes) - set(gold_amr.alignments)
    unaligned_nodes |= \
        set(nid for nid, pos in gold_amr.alignments.items() if pos is None)
    unaligned_nodes = sorted(list(unaligned_nodes))
    unaligned_nodes_original = sorted(list(unaligned_nodes))

    if not unaligned_nodes:
        # no need to do anything
        return gold_amr, []

    if len(unaligned_nodes) == 1 and len(gold_amr.tokens) == 1:
        # Degenerate case: single token
        node_id = list(unaligned_nodes)[0]
        gold_amr.alignments[node_id] = [0]
        return gold_amr, []

    # Align unaligned nodes by using graph vicinnity greedily (1 hop at a time)
    while unaligned_nodes:
        fix_alignments = graph_alignments(unaligned_nodes, gold_amr)
        for nid in unaligned_nodes:
            if nid in fix_alignments:
                gold_amr.alignments[nid] = [fix_alignments[nid]]
                unaligned_nodes.remove(nid)

        # debug: avoid infinite loop for AMR2.0 test data with bad alignments
        if not fix_alignments:
            # breakpoint()
            print(red_background('hard fix on 0th token for fix_alignments'))
            for k, v in list(gold_amr.alignments.items()):
                if v is None:
                    gold_amr.alignments[k] = [0]
            break

    return gold_amr, unaligned_nodes_original


def normalize(token):
    """
    Normalize token or node
    """
    if token == '"':
        return token
    else:
        return token.replace('"', '')


class AMROracle():

    def __init__(self, reduce_nodes=None, absolute_stack_pos=False,
                 use_copy=True):

        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to end of self.node_stack
        self.absolute_stack_pos = absolute_stack_pos

        # use copy action
        self.use_copy = use_copy

    def reset(self, gold_amr):

        # Force align missing nodes and store names for stats
        self.gold_amr, self.unaligned_nodes = fix_alignments(gold_amr)

        # will store alignments by token
        # TODO: This should store alignment probabilities
        align_by_token_pos = defaultdict(list)
        for node_id, token_pos in self.gold_amr.alignments.items():
            node = normalize(self.gold_amr.nodes[node_id])
            matched = False
            for pos in token_pos:
                if node == self.gold_amr.tokens[pos]:
                    align_by_token_pos[pos].append(node_id)
                    matched = True
            if not matched:
                align_by_token_pos[token_pos[0]].append(node_id)
        self.align_by_token_pos = align_by_token_pos

        node_id_2_node_number = {}
        for token_pos in sorted(self.align_by_token_pos.keys()):
            for node_id in self.align_by_token_pos[token_pos]:
                node_number = len(node_id_2_node_number)
                node_id_2_node_number[node_id] = node_number

        # will store edges not yet predicted indexed by node
        self.pend_edges_by_node = defaultdict(list)
        for (src, label, tgt) in self.gold_amr.edges:
            self.pend_edges_by_node[src].append((src, label, tgt))
            self.pend_edges_by_node[tgt].append((src, label, tgt))

        # sort edges in descending order of node2pos position
        for node_id in self.pend_edges_by_node:
            edges = []
            for (idx, e) in enumerate(self.pend_edges_by_node[node_id]):
                other_id = e[0]
                if other_id == node_id:
                    other_id = e[2]
                edges.append((node_id_2_node_number[other_id], idx))
            edges.sort(reverse=True)
            new_edges_for_node = []
            for (_, idx) in edges:
                new_edges_for_node.append(
                    self.pend_edges_by_node[node_id][idx])
            self.pend_edges_by_node[node_id] = new_edges_for_node

        # Will store gold_amr.nodes.keys() and edges as we predict them
        self.node_map = {}
        self.node_reverse_map = {}
        self.predicted_edges = []

    def get_arc_action(self, machine):

        # Loop over edges not yet created
        top_node_id = machine.node_stack[-1]
        current_id = self.node_reverse_map[top_node_id]
        for (src, label, tgt) in self.pend_edges_by_node[current_id]:
            # skip if it involves nodes not yet created
            if src not in self.node_map or tgt not in self.node_map:
                continue
            if (
                self.node_map[src] == top_node_id
                and self.node_map[tgt] in machine.node_stack[:-1]
            ):
                # LA <--
                if self.absolute_stack_pos:
                    # node position is just position in action history
                    index = self.node_map[tgt]
                else:
                    # stack position 0 is closest to current node
                    index = machine.node_stack.index(self.node_map[tgt])
                    index = len(machine.node_stack) - index - 2
                # Remove this edge from for both involved nodes
                self.pend_edges_by_node[tgt].remove((src, label, tgt))
                self.pend_edges_by_node[current_id].remove((src, label, tgt))
                # return [f'LA({label[1:]};{index})'], [1.0]
                # NOTE include the relation marker ':' in action names
                assert label[0] == ':'
                return [f'>LA({index},{label})'], [1.0]

            elif (
                self.node_map[tgt] == top_node_id
                and self.node_map[src] in machine.node_stack[:-1]
            ):
                # RA -->
                # note stack position 0 is closest to current node
                if self.absolute_stack_pos:
                    # node position is just position in action history
                    index = self.node_map[src]
                else:
                    # Relative node position
                    index = machine.node_stack.index(self.node_map[src])
                    index = len(machine.node_stack) - index - 2
                # Remove this edge from for both involved nodes
                self.pend_edges_by_node[src].remove((src, label, tgt))
                self.pend_edges_by_node[current_id].remove((src, label, tgt))
                # return [f'RA({label[1:]};{index})'], [1.0]
                # NOTE include the relation marker ':' in action names
                assert label[0] == ':'
                return [f'>RA({index},{label})'], [1.0]

    def get_reduce_action(self, machine, top=True):
        """
        If last action is an arc, check if any involved node (top or not top)
        has no pending edges
        """
        if machine.action_history == []:
            return False
        action = machine.action_history[-1]
        fetch = arc_regex.match(action)
        if fetch is None:
            return False
        if top:
            node_id = machine.node_stack[-1]
        else:
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                node_id = index
            else:
                # Relative node position
                index = len(machine.node_stack) - index - 2
                node_id = machine.node_stack[index]
        gold_node_id = self.node_reverse_map[node_id]
        return self.pend_edges_by_node[gold_node_id] == []

    def get_actions(self, machine):

        # Label node as root
        if (
            machine.node_stack
            and machine.root is None
            and self.node_reverse_map[machine.node_stack[-1]] ==
                self.gold_amr.root
        ):
            return ['ROOT'], [1.0]

        # REDUCE in stack after are LA/RA that completes all edges for an node
        if self.reduce_nodes == 'all':
            arc_reduce_no_top = self.get_reduce_action(machine, top=False)
            arc_reduce_top = self.get_reduce_action(machine, top=True)
            if arc_reduce_no_top and arc_reduce_top:
                # both nodes invoved
                return ['REDUCE3'], [1.0]
            elif arc_reduce_top:
                # top of the stack node
                return ['REDUCE'], [1.0]
            elif arc_reduce_no_top:
                # the other node
                return ['REDUCE2'], [1.0]

        # Return action creating next pending edge last node in stack
        if len(machine.node_stack) > 1:
            arc_action = self.get_arc_action(machine)
            if arc_action:
                return arc_action

        # Return action creating next node aligned to current cursor
        for nid in self.align_by_token_pos[machine.tok_cursor]:
            if nid in self.node_map:
                continue

            # NOTE: For PRED action we also include the gold id for
            # tracking and scoring of graph
            target_node = normalize(self.gold_amr.nodes[nid])

            if (
                self.use_copy and
                normalize(machine.tokens[machine.tok_cursor]) == target_node
            ):
                # COPY
                return [('COPY', nid)], [1.0]
            else:
                # Generate
                return [(target_node, nid)], [1.0]

        # Move monotonic attention
        if machine.tok_cursor < len(machine.tokens):
            return ['SHIFT'], [1.0]

        return ['CLOSE'], [1.0]


class AlignModeTracker():
    '''
    Tracks alignment of decoded AMR in align-mode to the corresponding gold AMR
    '''

    def __init__(self, gold_amr):

        self.gold_amr = gold_amr

        # assign gold node ids to decoded node ids based on node label
        gnode_by_label = defaultdict(list)
        for gnid, gnname in gold_amr.nodes.items():
            gnode_by_label[normalize(gnname)].append(gnid)

        # collect ambiguous and certain mappings separately in the structure
        # {node_label: [[gold_ids], [decoded_ids]]}
        self.gold_id_map = defaultdict(lambda: [[], []])
        self.ambiguous_gold_id_map = defaultdict(lambda: [[], []])
        for gnname, gnids in gnode_by_label.items():
            if len(gnids) == 1:
                self.gold_id_map[gnname] = [gnids, []]
            else:
                self.ambiguous_gold_id_map[gnname] = [gnids, []]

        # re-entrancy counts for both gold and decoded
        # {node_id: number_of_parents}
        self.num_gold_parents = {
            n: len(self.gold_amr.parents(n)) for n in self.gold_amr.nodes
        }
        self.num_dec_parents = {}

        # a data structure list(dict|list) that can disambiguate a decoded node
        # (assign it to it gold node, give sufficient decoded edges). This will
        # search entire graph neighbourhood
        self.dec_neighbours = get_gold_node_hashes(
            gold_amr.nodes, gold_amr.edges
        )
        # the same but given gold_ids already alignet to decoded ids. this only
        # needs size 1
        self.gold_neighbours = get_gold_node_hashes(
            gold_amr.nodes, gold_amr.edges, ids=True
        )

        # sanity check all nodes can be disambiguated with current conditions
        for nname in self.dec_neighbours:
            if (
                self.dec_neighbours[nname] == []
                and self.gold_neighbours[nname] == []
            ):
                set_trace(context=30)
                print()

        # there can be more than one edge betweentow nodes e.g. John hurt
        # himself
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

    def __str__(self):

        keys = [k for k, v in self.ambiguous_gold_id_map.items() if v[1]]
        string = ' '.join(f'{k} {self.gold_id_map[k]}' for k in keys)
        string += ' \n\n'
        string += ' '.join(
            f'{k} {self.ambiguous_gold_id_map[k]}' for k in keys
        )
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

    def get_flat_map(self, reverse=False, ambiguous=False):
        '''
        Get a flat map from gold_id_map and optionally ambiguous_gold_id_map
        relating gold nodes with decoded ones
        '''

        dec2gold = {}
        # Normal map is ordered and maps one to one
        for nname, (gnids, nids) in self.gold_id_map.items():
            for i in range(len(nids)):
                if reverse:
                    if ambiguous:
                        # TODO: Unify type here
                        dec2gold[gnids[i]] = [nids[i]]
                    else:
                        dec2gold[gnids[i]] = nids[i]
                else:
                    if ambiguous:
                        dec2gold[nids[i]] = [gnids[i]]
                    else:
                        dec2gold[nids[i]] = gnids[i]

        # Add also one to many map from ambiguous nodes
        if ambiguous:
            # Ambiguous maps have nor order, asign every gold to every decoded.
            for gnname, (gnids, nids) in self.ambiguous_gold_id_map.items():
                if nids != []:
                    if reverse:
                        for gnid in gnids:
                            dec2gold[gnid] = nids
                    else:
                        for nid in nids:
                            dec2gold[nid] = gnids

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
        for gnname, (gnids, nids) in self.ambiguous_gold_id_map.items():

            if nids == []:
                # already solved or no new nodes to disambiguate
                continue

            # disambiguate
            # FIXME: If the keys do not uniquely identify nodes, this will
            # be a greedy selection
            for nid in nids:

                matches = get_matching_gold_ids(
                    machine.nodes, machine.edges, nid,
                    self.dec_neighbours[gnname]
                )

                prop_matches = get_matching_gold_ids(
                    machine.nodes, machine.edges, nid,
                    self.gold_neighbours[gnname], id_map=self.get_flat_map()
                )
                if prop_matches and len(prop_matches) < len(matches):
                    matches = prop_matches

                # TODO: Support |subset| > 1 assignment
                if len(matches) == 1:
                    gnid = matches[0]
                    if nid not in self.gold_id_map[gnname][1]:
                        self.disambiguate_pair(gnname, gnid, nid)

    def update(self, machine):
        '''
        Update state of aligner given latest state machine state
        '''

        # note that actions predict node labels, not node ids so during partial
        # decoding we may not know which gold node corresponds to which decoded
        # node until a number of edges have been predicted.

        # add nodes created in this update
        dec2gold = self.get_flat_map(ambiguous=True)
        for nid, nname in machine.nodes.items():
            if nid not in dec2gold:
                if nname in self.ambiguous_gold_id_map:
                    self.ambiguous_gold_id_map[nname][1].append(nid)
                elif nname in self.gold_id_map:
                    # FIXME:
                    self.gold_id_map[nname][1].append(nid)
                else:
                    raise Exception()

        # update re-entrancy counts
        num_dec_parents = Counter()
        for (s, l, t) in machine.edges:
            if all(
                n != self.gold_amr.root
                and num_dec_parents[t] == self.num_gold_parents[n]
                for n in dec2gold[t]
            ):
                # at least one option should have less number of parents
                set_trace(context=30)
                print()
            num_dec_parents[t] += 1
        self.num_dec_parents = num_dec_parents

        # if machine.action_history and 'have-degree-91' in machine.action_history:
        # if machine.action_history and '>LA(43,:ARG3)' in machine.action_history:
        # if machine.action_history and '>LA(29,:ARG1-of)' in machine.action_history:
        #    print_and_break(30, self, machine)

        # update gold and decoded graph alignments
        # propagating from aligned pairs to unaligned ones
        # self._map_decoded_and_gold_ids_by_propagation(machine)
        # by context matching
        self._map_decoded_and_gold_ids_by_context(machine)

        # when only one ambiguous node is left, its not ambiguous any more
        for nname in self.ambiguous_gold_id_map.keys():
            self.exclusion_disambiguation(nname)

        # update edge alignments
        # TODO: maybe use node alignment here for propagation (iteratively)
        self._map_decoded_and_gold_edges(machine)

    def exclusion_disambiguation(self, nname):
        # if only one node left in ambiguous_gold_id_map, it is not ambiguous
        if (
            len(self.ambiguous_gold_id_map[nname][0]) == 1
            and len(self.ambiguous_gold_id_map[nname][1]) == 1
        ):
            gold_id = self.ambiguous_gold_id_map[nname][0][0]
            decoded_id = self.ambiguous_gold_id_map[nname][1][0]
            self.disambiguate_pair(nname, gold_id, decoded_id)

    def disambiguate_pair(self, nname, gold_id, decoded_id):

        # FIXME: Debug
        if gold_id in self.gold_id_map[nname][0]:
            set_trace(context=30)
            print()
        if gold_id not in self.ambiguous_gold_id_map[nname][0]:
            set_trace(context=30)
            print()

        # remove from ambiguous list
        self.ambiguous_gold_id_map[nname][0].remove(gold_id)
        self.ambiguous_gold_id_map[nname][1].remove(decoded_id)
        if self.ambiguous_gold_id_map[nname][0] == []:
            del self.ambiguous_gold_id_map[nname]

        # Add to final list
        self.gold_id_map[nname][0].append(gold_id)
        self.gold_id_map[nname][1].append(decoded_id)

        # if only one node left in ambiguous_gold_id_map, it is not ambiguous
        self.exclusion_disambiguation(nname)

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

        # if Counter(machine.action_history)['name'] > 1:
        # if machine.action_history and '>LA(7,:op1)' in machine.action_history:
        #    print_and_break(1, self, machine)

        # get a map from every gold node to every potential aligned decoded
        gold_to_dec_ids = self.get_flat_map(reverse=True, ambiguous=True)
        # get potential gold edges to predicty given stack content
        potential_gold_edges = self.get_potential_gold_edges(
            machine, gold_to_dec_ids
        )

        # note that during partial decoding we may have multiple possible
        # decoded edges for each gold edge (due to ambiguous node mapping above)
        # Here we cluster gold edges by same potential decoded edges.
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

            # store potential decodable edges for this gold edge. Note that we
            # can further disambiguate for this given gold edge
            for nid in gold_to_dec_ids[gold_s_id]:

                if nid in used_nids:
                    # if we used this already above, is not a possible decoding
                    continue
                for nid2 in gold_to_dec_ids[gold_t_id]:
                    if (
                        # we need to take into account re-entrancies
                        self.num_dec_parents[nid2] == self.num_gold_parents[gold_t_id]
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
        for nname, (gnids, nids) in self.gold_id_map.items():
            if len(gnids) > len(nids) and nname not in missing_gold_nodes:
                if repeat:
                    missing_gold_nodes.extend(
                        [nname] * (len(gnids) - len(nids))
                    )
                else:
                    missing_gold_nodes.append(nname)
        for nname, (gnids, nids) in self.ambiguous_gold_id_map.items():
            if len(gnids) > len(nids) and nname not in missing_gold_nodes:
                if repeat:
                    missing_gold_nodes.extend(
                        [nname] * (len(gnids) - len(nids))
                    )
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

        # expand to all possible disambiguations
        expanded_missing_gold_edges = []
        for gold_edge in missing_gold_edges:
            for gold_edge_candidate in self.goldedges_to_candidates[gold_edge]:
                expanded_missing_gold_edges.append(gold_edge_candidate)

        return expanded_missing_gold_edges


class AMRStateMachine():

    def __init__(self, reduce_nodes=None, absolute_stack_pos=False,
                 use_copy=True):

        # Here non state variables (do not change across sentences) as well as
        # slow initializations
        # Remove nodes that have all their edges created
        self.reduce_nodes = reduce_nodes
        # e.g. LA(<label>, <pos>) <pos> is absolute position in sentence,
        # rather than relative to stack top
        self.absolute_stack_pos = absolute_stack_pos

        # use copy action
        self.use_copy = use_copy

        # base actions allowed
        self.base_action_vocabulary = [
            'SHIFT',   # Move cursor
            'COPY',    # Copy word under cursor to node (add node to stack)
            'ROOT',    # Label node as root
            # Arc from node under cursor (<label>, <to position>) (to be
            # different from LA the city)
            '>LA',
            '>RA',      # Arc to node under cursor (<label>, <from position>)
            'CLOSE',   # Close machine
            # ...      # create node with ... as name (add node to stack)
            'NODE'     # other node names
        ]
        if self.reduce_nodes:
            self.base_action_vocabulary.append([
                'REDUCE',   # Remove node at top of the stack
                'REDUCE2',  # Remove node at las LA/RA pointed position
                'REDUCE3'   # Do both above
            ])

        if not self.use_copy:
            self.base_action_vocabulary.remove('COPY')

    def canonical_action_to_dict(self, vocab):
        """
        Map the canonical actions to ids in a vocabulary, each canonical action
        corresponds to a set of ids.

        CLOSE is mapped to eos </s> token.
        """
        canonical_act_ids = dict()
        vocab_act_count = 0
        assert vocab.eos_word == '</s>'
        for i in range(len(vocab)):
            # NOTE can not directly use "for act in vocab" -> this will never
            # stop since no stopping iter implemented
            act = vocab[i]
            if (
                act in ['<s>', '<pad>', '<unk>', '<mask>']
                or act.startswith('madeupword')
            ):
                continue
            cano_act = self.get_base_action(
                act) if i != vocab.eos() else 'CLOSE'
            if cano_act in self.base_action_vocabulary:
                vocab_act_count += 1
                canonical_act_ids.setdefault(cano_act, []).append(i)
        # print for debugging
        # print(f'{vocab_act_count} / {len(vocab)} tokens in action vocabulary
        # mapped to canonical actions.')
        return canonical_act_ids

    def reset(self, tokens, gold_amr=None):
        '''
        Reset state variables and set a new sentence

        Use gold_amr for align mode
        '''
        # state
        self.tokens = list(tokens)
        self.tok_cursor = 0
        self.node_stack = []
        self.action_history = []

        # AMR as we construct it
        # NOTE: We will use position of node generating action in action
        # history as node_id
        self.nodes = {}
        self.edges = []
        self.root = None
        self.alignments = defaultdict(list)
        # set to true when machine finishes
        self.is_closed = False

        # state info useful in the model
        self.actions_tokcursor = []

        # align mode
        self.gold_amr = gold_amr
        if gold_amr:
            # this will track the node alignments between
            self.align_tracker = AlignModeTracker(gold_amr)
            self.align_tracker.update(self)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as fid:
            config = json.loads(fid.read())
        return cls(**config)

    def save(self, config_path):
        with open(config_path, 'w') as fid:
            # NOTE: Add here all *non state* variables in __init__()
            fid.write(json.dumps(dict(
                reduce_nodes=self.reduce_nodes,
                absolute_stack_pos=self.absolute_stack_pos,
                use_copy=self.use_copy
            )))

    def __deepcopy__(self, memo):
        """
        Manual deep copy of the machine

        avoid deep copying heavy files
        """
        cls = self.__class__
        result = cls.__new__(cls)
        # DEBUG: usew this to detect very heavy constants that can be referred
        # import time
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # start = time.time()
            # if k in ['actions_by_stack_rules']:
            #     setattr(result, k, v)
            # else:
            #     setattr(result, k, deepcopy(v, memo))
            setattr(result, k, deepcopy(v, memo))
            # print(k, time.time() - start)
        # import ipdb; ipdb.set_trace(context=30)
        return result

    def state_str(self, node_map=None):
        '''
        Return string representing machine state
        '''
        string = ' '.join(self.tokens[:self.tok_cursor])
        if self.tok_cursor < len(self.tokens):
            string += f' \033[7m{self.tokens[self.tok_cursor]}\033[0m '
            string += ' '.join(self.tokens[self.tok_cursor+1:]) + '\n\n'
        else:
            string += '\n\n'

        string += ' '.join(self.action_history) + '\n\n'
        if self.edges:
            amr_str = self.get_amr().to_penman(node_map=node_map)
        else:
            # invalid AMR
            amr_str = '\n'.join(
                f'({nid} / {nname})' for nid, nname in self.nodes.items()
            )
        amr_str = '\n'.join(
            x for x in amr_str.split('\n') if x and x[0] != '#'
        )
        string += f'{amr_str}\n\n'

        return string

    def __str__(self):
        return self.state_str()

    def get_current_token(self):
        if self.tok_cursor >= len(self.tokens):
            return None
        else:
            return self.tokens[self.tok_cursor]

    def get_base_action(self, action):
        """Get the base action form, by stripping the labels, etc."""
        if action in self.base_action_vocabulary:
            return action
        # remaining ones are ['>LA', '>RA', 'NODE']
        # NOTE need to deal with both '>LA(pos,label)' and '>LA(label)', as in
        # the vocabulary the pointers are peeled off
        if arc_regex.match(action) or arc_nopointer_regex.match(action):
            return action[:3]
        return 'NODE'

    def _get_valid_align_actions(self):
        '''Get actions that generate given gold AMR'''

        # if self.action_history and '>RA(38,:mod)' in self.action_history:
        #    print_and_break(30, self.align_tracker, self)

        # return arc actions if any
        # corresponding possible decoded edges
        arc_actions = []
        for (s, gold_e_label, t) in self.align_tracker.get_missing_edges(self):
            if s in self.node_stack[:-1] and t == self.node_stack[-1]:
                # right arc stack --> top
                action = f'>RA({s},{gold_e_label})'
            else:
                # left arc stack <-- top
                action = f'>LA({t},{gold_e_label})'
            if action not in arc_actions:
                arc_actions.append(action)

        if arc_actions:
            # TODO: Pointer and label can only be enforced independently, which
            # means that if we hae two diffrent arcs to choose from, we could
            # make a mistake. We need to enforce an arc order.
            return arc_actions

        # otherwise choose between producing a gold node and shifting (if
        # possible)
        valid_base_actions = []
        for nname in self.align_tracker.get_missing_nnames():
            if normalize(nname) == self.get_current_token():
                valid_base_actions.append('COPY')
            else:
                valid_base_actions.append(normalize(nname))
        if self.tok_cursor < len(self.tokens):
            valid_base_actions.append('SHIFT')

        if valid_base_actions == []:
            # if no possible option, just close
            return ['CLOSE']
        else:
            return valid_base_actions

    def get_valid_actions(self, max_1root=True):

        if self.gold_amr:
            # align mode (we know the AMR)
            return self._get_valid_align_actions()

        valid_base_actions = []
        gen_node_actions = ['COPY', 'NODE'] if self.use_copy else ['NODE']

        if self.tok_cursor < len(self.tokens):
            valid_base_actions.append('SHIFT')
            valid_base_actions.extend(gen_node_actions)

        if (
            self.action_history
            and self.get_base_action(self.action_history[-1]) in (
                gen_node_actions + ['ROOT', '>LA', '>RA']
            )
        ):
            valid_base_actions.extend(['>LA', '>RA'])

        if (
            self.action_history
            and self.get_base_action(self.action_history[-1])
                in gen_node_actions
        ):
            if max_1root:
                # force to have at most 1 root (but it can always be with no
                # root)
                if not self.root:
                    valid_base_actions.append('ROOT')
            else:
                valid_base_actions.append('ROOT')

        if self.tok_cursor == len(self.tokens):
            assert not valid_base_actions \
                and self.action_history[-1] == 'SHIFT'
            valid_base_actions.append('CLOSE')

        if self.reduce_nodes:
            raise NotImplementedError

            if len(self.node_stack) > 0:
                valid_base_actions.append('REDUCE')
            if len(self.node_stack) > 1:
                valid_base_actions.append('REDUCE2')
                valid_base_actions.append('REDUCE3')

        return valid_base_actions

    def get_actions_nodemask(self):
        """Get the binary mask of node actions"""
        actions_nodemask = [0] * len(self.action_history)
        for i in self.node_stack:
            actions_nodemask[i] = 1
        return actions_nodemask

    def update(self, action):

        assert not self.is_closed

        # FIXME: Align mode can not allow '<unk>' node names but we need a
        # handling of '<unk>' that works with other NN vocabularies
        if self.gold_amr and action == '<unk>':
            valid_actions = ' '.join(self.get_valid_actions())
            raise Exception(
                f'{valid_actions} is an <unk> action: you can not use align '
                'mode enforcing actions not in the vocabulary'
            )

        self.actions_tokcursor.append(self.tok_cursor)

        if re.match(r'CLOSE', action):
            self.is_closed = True

        elif re.match(r'ROOT', action):
            self.root = self.node_stack[-1]

        elif action in ['SHIFT']:
            # Move source pointer
            self.tok_cursor += 1

        elif action in ['REDUCE']:
            # eliminate top of the stack
            assert self.reduce_nodes
            assert self.action_history[-1]
            self.node_stack.pop()

        elif action in ['REDUCE2']:
            # eliminate the other node involved in last arc not on top
            assert self.reduce_nodes
            assert self.action_history[-1]
            fetch = arc_regex.match(self.action_history[-1])
            assert fetch
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                # Absolute position and also node_id
                self.node_stack.remove(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                self.node_stack.pop(index)

        elif action in ['REDUCE3']:
            # eliminate both nodes involved in arc
            assert self.reduce_nodes
            assert self.action_history[-1]
            fetch = arc_regex.match(self.action_history[-1])
            assert fetch
            # index = int(fetch.groups()[1])
            index = int(fetch.groups()[0])
            if self.absolute_stack_pos:
                # Absolute position and also node_id
                self.node_stack.remove(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                self.node_stack.pop(index)
            self.node_stack.pop()

        # Edge generation
        elif la_regex.match(action):
            # Left Arc <--
            # label, index = la_regex.match(action).groups()
            index, label = la_regex.match(action).groups()
            if self.absolute_stack_pos:
                tgt = int(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                tgt = self.node_stack[index]
            src = self.node_stack[-1]
            self.edges.append((src, f'{label}', tgt))

        elif ra_regex.match(action):
            # Right Arc -->
            # label, index = ra_regex.match(action).groups()
            index, label = ra_regex.match(action).groups()
            if self.absolute_stack_pos:
                src = int(index)
            else:
                # Relative position
                index = len(self.node_stack) - int(index) - 2
                src = self.node_stack[index]
            tgt = self.node_stack[-1]
            self.edges.append((src, f'{label}', tgt))

        # Node generation
        elif action == 'COPY':
            # copy surface symbol under cursor to node-name
            node_id = len(self.action_history)
            self.nodes[node_id] = normalize(self.tokens[self.tok_cursor])
            self.node_stack.append(node_id)
            self.alignments[node_id].append(self.tok_cursor)

        else:

            # Interpret action as a node name
            # Note that the node_id is the position of the action that
            # generated it
            node_id = len(self.action_history)
            self.nodes[node_id] = action
            self.node_stack.append(node_id)
            self.alignments[node_id].append(self.tok_cursor)

        # Action for each time-step
        self.action_history.append(action)

        # Update align mode tracker after machine state has been updated
        if self.gold_amr:
            self.align_tracker.update(self)

    def get_amr(self):
        return AMR(self.tokens, self.nodes, self.edges, self.root,
                   alignments=self.alignments, clean=True, connect=True)

    def get_aligned_amr(self):

        # special handling for align mode
        # TODO: Just alter self.gold_amr.penman alignments for max
        # compatibility

        # map from decoded nodes to gold nodes
        gold2dec = self.align_tracker.get_flat_map(reverse=True)

        if self.root is None:
            # FIXME: This is because ROOT is predicted in-situ and can not
            # be used in align mode (we do not allways know root in-situ)
            # this should be fixable with non in-situ root
            if self.gold_amr.root in gold2dec:
                # this will not work for partial AMRs
                self.root = gold2dec[self.gold_amr.root]

        return AMR(self.tokens, self.nodes, self.edges, self.root,
                   alignments=self.alignments, clean=True, connect=True)

    def get_annotation(self, node_map=None):
        if self.gold_amr:

            # DEBUG
            # debug_align_mode(self)

            # Align mode
            # If we read the gold AMR from penman, we can just apply the
            # alignments to it, thus keeping the epidata info intact
            if self.gold_amr.penman:
                gold2dec = self.align_tracker.get_flat_map()
                alignments = {
                    gold2dec[nid]: pos for nid, pos in self.alignments.items()
                }
                return add_alignments_to_penman(
                    self.gold_amr.penman,
                    alignments,
                    string=True
                )

            else:
                node_map = self.align_tracker.get_flat_map()
                return self.get_aligned_amr().to_penman(node_map=node_map)
        else:
            return self.get_amr().to_jamr()


def get_ngram(sequence, order):
    ngrams = []
    for n in range(len(sequence) - order + 1):
        ngrams.append(tuple(sequence[n:n+order]))
    return ngrams


class Stats():

    def __init__(self, ignore_indices, ngram_stats=False, breakpoint=False):
        self.index = 0
        self.ignore_indices = ignore_indices
        # arc generation stats
        self.stack_size_count = Counter()
        self.pointer_positions_count = Counter()
        # alignment stats
        self.unaligned_node_count = Counter()
        self.node_count = 0
        # token/action stats
        self.tokens = []
        self.action_sequences = []
        self.action_count = Counter()

        self.ngram_stats = ngram_stats
        self.breakpoint = breakpoint

        # Stats for action n-grams
        if self.ngram_stats:
            self.bigram_count = Counter()
            self.trigram_count = Counter()
            self.fourgram_count = Counter()

    def update_machine_stats(self, machine):

        if self.breakpoint:
            os.system('clear')
            print(" ".join(machine.tokens))
            print(" ".join(machine.action_history))
            print(" ".join([machine.action_history[i]
                            for i in machine.node_stack]))
            set_trace()

            # if len(machine.node_stack) > 8 and stats.index not in [12]:
            #    set_trace(context=30)

        action = machine.action_history[-1]
        fetch = arc_regex.match(action)
        if fetch:
            # stack_pos = int(fetch.groups()[1])
            stack_pos = int(fetch.groups()[0])
            self.stack_size_count.update([len(machine.node_stack)])
            self.pointer_positions_count.update([stack_pos])

    def update_sentence_stats(self, oracle, machine):

        # Note that we do not ignore this one either way
        self.tokens.append(machine.tokens)
        self.action_sequences.append(machine.action_history)
        base_actions = [x.split('(')[0] for x in machine.action_history]
        self.action_count.update(base_actions)

        # alignment fixing stats
        unodes = [oracle.gold_amr.nodes[n] for n in oracle.unaligned_nodes]
        self.unaligned_node_count.update(unodes)
        self.node_count += len(oracle.gold_amr.nodes)

        if self.index in self.ignore_indices:
            self.index += 1
            return

        if self.ngram_stats:
            actions = machine.action_history
            self.bigram_count.update(get_ngram(actions, 2))
            self.trigram_count.update(get_ngram(actions, 3))
            self.fourgram_count.update(get_ngram(actions, 4))

        # breakpoint if AMR does not match
        self.stop_if_error(oracle, machine)

        # update counter
        self.index += 1

    def stop_if_error(self, oracle, machine):

        # Check node name match
        for nid, node_name in oracle.gold_amr.nodes.items():
            node_name_machine = machine.nodes[oracle.node_map[nid]]
            if normalize(node_name_machine) != normalize(node_name):
                set_trace(context=30)
                print()

        # Check mapped edges match
        mapped_gold_edges = []
        for (s, label, t) in oracle.gold_amr.edges:
            if s not in oracle.node_map or t not in oracle.node_map:
                set_trace(context=30)
                continue
            mapped_gold_edges.append(
                (oracle.node_map[s], label, oracle.node_map[t])
            )
        if sorted(machine.edges) != sorted(mapped_gold_edges):
            set_trace(context=30)
            print()

        # Check root matches
        mapped_root = oracle.node_map[oracle.gold_amr.root]
        if machine.root != mapped_root:
            set_trace(context=30)
            print()

    def display(self):

        num_processed = self.index - len(self.ignore_indices)
        perc = num_processed * 100. / self.index
        print(
            f'{num_processed}/{self.index} ({perc:.1f} %)'
            f' exact match of AMR graphs (non printed)'
        )
        print(yellow_font(
            f'{len(self.ignore_indices)} sentences ignored for stats'
        ))

        num_copy = self.action_count['COPY']
        perc = num_copy * 100. / self.node_count
        print(
            f'{num_copy}/{self.node_count} ({perc:.1f} %) of nodes generated'
            ' by COPY'
        )

        if self.unaligned_node_count:
            num_unaligned = sum(self.unaligned_node_count.values())
            print(yellow_font(
                f'{num_unaligned}/{self.node_count} unaligned nodes aligned'
                ' by graph vicinity'
            ))

        # Other stats
        return

        # format viewer
        clbar2 = partial(clbar, ylim=(0, None), norm=True,
                         yform=lambda y: f'{100*y:.1f}', ncol=80)

        print('Stack size')
        clbar2(xy=self.stack_size_count, botx=20)

        print('Positions')
        clbar2(xy=self.pointer_positions_count, botx=20)

        if self.ngram_stats:
            print('tri-grams')
            clbar(xy=self.trigram_count, topy=20)
            set_trace()
            print()


def peel_pointer(action, pad=-1):
    """Peel off the pointer value from arc actions"""
    if arc_regex.match(action):
        # LA(pos,label) or RA(pos,label)
        action, properties = action.split('(')
        properties = properties[:-1]    # remove the ')' at last position
        # split to pointer value and label
        properties = properties.split(',')
        pos = int(properties[0].strip())
        # remove any leading and trailing white spaces
        label = properties[1].strip()
        action_label = action + '(' + label + ')'
        return (action_label, pos)
    else:
        return (action, pad)


class StatsForVocab:
    """
    Collate stats for predicate node names with their frequency, and list of
    all the other action symbols.  For arc actions, pointers values are
    stripped.  The results stats (from training data) are going to decide which
    node names (the frequent ones) to be added to the vocabulary used in the
    model.
    """

    def __init__(self, no_close=False):
        # DO NOT include CLOSE action (as this is internally managed by the eos
        # token in model)
        # NOTE we still add CLOSE into vocabulary, just to be complete although
        # it is not used
        self.no_close = no_close

        self.nodes = Counter()
        self.left_arcs = Counter()
        self.right_arcs = Counter()
        self.control = Counter()

    def update(self, action, machine):
        if self.no_close:
            if action in ['CLOSE', '_CLOSE_']:
                return

        if la_regex.match(action) or la_nopointer_regex.match(action):
            # LA(pos,label) or LA(label)
            action, pos = peel_pointer(action)
            # NOTE should be an iterable instead of a string; otherwise it'll
            # be character based
            self.left_arcs.update([action])
        elif ra_regex.match(action) or ra_nopointer_regex.match(action):
            # RA(pos,label) or RA(label)
            action, pos = peel_pointer(action)
            self.right_arcs.update([action])
        elif action in machine.base_action_vocabulary:
            self.control.update([action])
        else:
            # node names
            self.nodes.update([action])

    def display(self):
        print('Total number of different node names:')
        print(len(list(self.nodes.keys())))
        print('Most frequent node names:')
        print(self.nodes.most_common(20))
        print('Most frequent left arc actions:')
        print(self.left_arcs.most_common(20))
        print('Most frequent right arc actions:')
        print(self.right_arcs.most_common(20))
        print('Other control actions:')
        print(self.control)

    def write(self, path_prefix):
        """
        Write the stats into file. Two files will be written: one for nodes,
        one for others.
        """
        path_nodes = path_prefix + '.nodes'
        path_others = path_prefix + '.others'
        with open(path_nodes, 'w') as f:
            for k, v in self.nodes.most_common():
                print(f'{k}\t{v}', file=f)
        with open(path_others, 'w') as f:
            for k, v in chain(
                self.control.most_common(),
                self.left_arcs.most_common(),
                self.right_arcs.most_common()
            ):
                print(f'{k}\t{v}', file=f)


def oracle(args):

    # Read AMR
    amrs = read_amr(args.in_aligned_amr, ibm_format=True)

    # broken annotations that we ignore in stats
    # 'DATA/AMR2.0/aligned/cofill/train.txt'
    ignore_indices = [
        8372,   # (49, ':time', 49), (49, ':condition', 49)
        17055,  # (3, ':mod', 7), (3, ':mod', 7)
        27076,  # '0.0.2.1.0.0' is on ::edges but not ::nodes
        # for AMR 3.0 data: DATA/AMR3.0/aligned/cofill/train.txt
        # self-loop:
        # "# ::edge vote-01 condition vote-01 0.0.2 0.0.2",
        # "# ::edge vote-01 time vote-01 0.0.2 0.0.2"
        9296,
    ]
    # NOTE we add indices to ignore for both amr2.0 and amr3.0 in the same list
    # and used for both oracles, since: this would NOT change the oracle
    # actions, but only ignore sanity checks and displayed stats after oracle
    # run

    # Initialize machine
    machine = AMRStateMachine(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions,
        use_copy=args.use_copy
    )
    # Save machine config
    machine.save(args.out_machine_config)

    # initialize oracle
    oracle = AMROracle(
        reduce_nodes=args.reduce_nodes,
        absolute_stack_pos=args.absolute_stack_positions,
        use_copy=args.use_copy
    )

    # will store statistics and check AMR is recovered
    stats = Stats(ignore_indices, ngram_stats=False)
    stats_vocab = StatsForVocab(no_close=False)
    for idx, amr in tqdm(enumerate(amrs), desc='Oracle'):

        # debug
        # print(idx)    # 96 for AMR2.0 test data infinit loop
        # if idx == 96:
        #     breakpoint()

        # spawn new machine for this sentence
        machine.reset(amr.tokens)

        # initialize new oracle for this AMR
        oracle.reset(amr)

        # proceed left to right throught the sentence generating nodes
        while not machine.is_closed:

            # get valid actions
            _ = machine.get_valid_actions()

            # oracle
            actions, scores = oracle.get_actions(machine)
            # actions = [a for a in actions if a in valid_actions]
            # most probable
            action = actions[np.argmax(scores)]

            # if it is node generation, keep track of original id in gold amr
            if isinstance(action, tuple):
                action, gold_node_id = action
                node_id = len(machine.action_history)
                oracle.node_map[gold_node_id] = node_id
                oracle.node_reverse_map[node_id] = gold_node_id

            # update machine,
            machine.update(action)

            # update machine stats
            stats.update_machine_stats(machine)

            # update vocabulary
            stats_vocab.update(action, machine)

        # Sanity check: We recovered the full AMR
        stats.update_sentence_stats(oracle, machine)

        # do not write 'CLOSE' in the action sequences
        # this might change the machine.action_history in place, but it is the
        # end of this machine already
        close_action = stats.action_sequences[-1].pop()
        assert close_action == 'CLOSE'

    # display statistics
    stats.display()

    # save action sequences and tokens
    write_tokenized_sentences(
        args.out_actions,
        stats.action_sequences,
        '\t'
    )
    write_tokenized_sentences(
        args.out_tokens,
        stats.tokens,
        '\t'
    )

    # save action vocabulary stats
    # debug

    stats_vocab.display()
    if getattr(args, 'out_stats_vocab', None) is not None:
        stats_vocab.write(args.out_stats_vocab)
        print(f'Action vocabulary stats written in {args.out_stats_vocab}.*')


def play(args):

    sentences = read_tokenized_sentences(args.in_tokens, '\t')
    action_sequences = read_tokenized_sentences(args.in_actions, '\t')
    assert len(sentences) == len(action_sequences)

    # This will store the annotations to write
    annotations = []

    # Initialize machine
    machine = AMRStateMachine.from_config(args.in_machine_config)
    for index in tqdm(range(len(action_sequences)), desc='Machine'):

        # New machine for this sentence
        machine.reset(sentences[index])

        # add back the 'CLOSE' action if it is not written in file
        if action_sequences[index][-1] != 'CLOSE':
            action_sequences[index].append('CLOSE')

        for action in action_sequences[index]:
            machine.update(action)

        assert machine.is_closed

        # print AMR
        annotations.append(machine.get_annotation())

    with open(args.out_amr, 'w') as fid:
        for annotation in annotations:
            fid.write(annotation)


def main(args):

    # TODO: Use two separate entry points with own argparse and requires

    if args.in_actions:
        # play actions and return AMR
        assert args.in_machine_config
        assert args.in_tokens
        assert args.in_actions
        assert args.out_amr
        assert not args.out_actions
        assert not args.in_aligned_amr
        play(args)

    elif args.in_aligned_amr:
        # Run oracle and determine actions from AMR
        assert args.in_aligned_amr
        assert args.out_actions
        assert args.out_tokens
        assert args.out_machine_config
        assert not args.in_tokens
        assert not args.in_actions
        oracle(args)


def argument_parser():

    parser = argparse.ArgumentParser(description='Aligns AMR to its sentence')
    # Single input parameters
    parser.add_argument(
        "--in-aligned-amr",
        help="In file containing AMR in penman format AND IBM graph notation "
             "(::node, etc). Graph read from the latter and not penman",
        type=str
    )
    # ORACLE
    parser.add_argument(
        "--reduce-nodes",
        choices=['all'],
        help="Rules to delete completed nodes during parsing"
             "all: delete every complete node",
        type=str,
    )
    parser.add_argument(
        "--absolute-stack-positions",
        help="e.g. LA(<label>, <pos>) <pos> is absolute position in sentence",
        action='store_true'
    )
    parser.add_argument(
        "--use-copy",
        help='Use COPY action to copy words at source token cursor',
        type=int,
    )
    parser.add_argument(
        "--out-actions",
        help="tab separated actions, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-tokens",
        help="space separated tokens, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-machine-config",
        help="configuration for state machine in config format",
        type=str,
    )
    parser.add_argument(
        "--out-stats-vocab",
        type=str,
        help="action vocabulary frequencies"
    )
    # MACHINE
    parser.add_argument(
        "--in-tokens",
        help="space separated tokens, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--in-actions",
        help="tab separated actions, one sentence per line",
        type=str,
    )
    parser.add_argument(
        "--out-amr",
        help="In file containing AMR in penman format",
        type=str,
    )
    parser.add_argument(
        "--in-machine-config",
        help="configuration for state machine in config format",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(argument_parser())
