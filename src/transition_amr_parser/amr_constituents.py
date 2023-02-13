from collections import defaultdict
from ipdb import set_trace


def get_reentrant_edges(amr, alignment_sort=False):
    '''
    Get re-entrant edges i.e. extra parents. We keep the edge closest to
    root or the edge with child aligned to the earliest token
    '''

    if alignment_sort:
        # annotate token position at which edge occurs
        depth_by_edge = dict()
        for (src, label, tgt) in amr.edges:
            if amr.alignments.get(src, False):
                depth_by_edge[(src, label, tgt)] = min(amr.alignments[src])
            else:
                depth_by_edge[(src, label, tgt)] = 10000
    else:
        # annotate depth at which edge occurs
        candidates = [amr.root]
        depths = [0]
        depth_by_edge = dict()
        while candidates:
            for (tgt, label) in amr.children(candidates[0]):
                edge = (candidates[0], label, tgt)
                if edge in depth_by_edge:
                    continue
                depth_by_edge[edge] = depths[0]
                candidates.append(tgt)
                depths.append(depths[0] + 1)
            candidates.pop(0)
            depths.pop(0)

    # in case of multiple parents keep the one closest to the root
    reentrancy_edges = []
    for nid, nname in amr.nodes.items():
        parents = [(src, label, nid) for src, label in amr.parents(nid)]
        if nid == amr.root:
            # Root can not have parents
            reentrancy_edges.extend(parents)
        elif len(parents) > 1:
            # Keep only highest edge from re-entrant ones
            # FIXME: Unclear why depth is missing sometimes
            reentrancy_edges.extend(
                sorted(parents, key=lambda e: depth_by_edge.get(e, 1000))[1:]
            )
    return reentrancy_edges


class DFS():
    '''
    Depth First Search in AMR graph handling re-entrancies
    '''

    def __init__(
        self,
        no_reentrancies=True,    # do not follow re-entrancies (loops otherw.)
        no_reverse_edges=False,  # do not follow e.g. ARG0-of
        alignment_sort=True      # determine first role and re-entrancis by
                                 # token positions or node positions
    ):
        self.no_reentrancies = no_reentrancies
        self.no_reverse_edges = no_reverse_edges
        self.alignment_sort = alignment_sort

    def reset(self, amr):
        # initialize search
        self.amr = amr

    def downwards(self, path, edge):
        # when going down one edge
        pass

    def upwards(self, path, child):
        # when coming up one edge
        pass

    def trasverse(self):

        # get re-entrant edges
        self.reentrant_edges = get_reentrant_edges(
            self.amr,
            alignment_sort=self.alignment_sort,
        )

        # transverse graph depth first search by following first child and
        # adding edge to a blocklist
        path = [self.amr.root]
        visited_edges = []
        while path:
            # get the next child reachable via a valid edge
            new_child = None
            for child, label in self.amr.children(path[-1]):
                if (
                    (path[-1], label, child) in visited_edges
                    or (
                        self.no_reverse_edges
                        and label.endswith('-of')
                    )
                ):
                    continue
                visited_edges.append((path[-1], label, child))
                new_child = child
                break

            if new_child:
                # downwards action
                # store this node for each ancestor
                edge = (path[-1], label, new_child)
                self.downwards(path, edge)
                # add as next if its not re-entrant
                if edge not in self.reentrant_edges:
                    path.append(new_child)
            else:
                # reached leaf-node, remove it from list
                pop_child = path.pop()
                # upwards action
                self.upwards(path, pop_child)


class SubGraph(DFS):
    # find all nodes below each node in a graph

    def reset(self, amr):
        self.amr = amr
        self.subgraph_by_id = defaultdict(set)
        self.is_reentrant_by_id = defaultdict(set)

    def downwards(self, path, edge):
        for n in path:
            if edge in self.reentrant_edges:
                self.is_reentrant_by_id[n] |= set([edge[2]])
            self.subgraph_by_id[n] |= set([edge[2]])


class NodeDepth(DFS):
    # Find the grid to plot an AMR graph (which nodes at which depth)

    def reset(self, amr):
        self.amr = amr
        self.subgraph_by_id = defaultdict(set)
        self.grid = defaultdict(list)

    def upwards(self, path, child):
        self.grid[len(path)].append(child)


def get_subgraph_by_id(amr, no_reentrancies=True, no_reverse_edges=False,
                       alignment_sort=True):

    # subgraph = NodeDepth(no_reentrancies=no_reentrancies,
    subgraph = SubGraph(no_reentrancies=no_reentrancies,
                        no_reverse_edges=no_reverse_edges,
                        alignment_sort=alignment_sort)
    subgraph.reset(amr)
    subgraph.trasverse()

    return subgraph.subgraph_by_id, subgraph.is_reentrant_by_id
