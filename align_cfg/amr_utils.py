import collections

from transition_amr_parser.io import read_amr2

import torch


class Corpus:
    pass


def read_amr(*args, **kwargs):
    amrs = read_amr2(*args, **kwargs)
    corpus = Corpus()
    corpus.amrs = amrs
    return corpus


def get_node_ids(amr):
    return list(sorted(amr.nodes.keys()))


def convert_amr_to_tree(amr):
    seen = set()

    # Note: If we skip adding the root, then cycles may form.
    seen.add(amr.root)

    tree = {}
    tree['root'] = amr.root
    tree['node_to_children'] = collections.defaultdict(list)
    tree['edge_to_label'] = {}
    tree['edges'] = []
    tree['node_ids'] = get_node_ids(amr)

    def sortkey(x):
        s, y, t = x

        return (s, t)

    for e in sorted(amr.edges, key=sortkey):
        s, y, t = e
        if t in seen:
            continue
        seen.add(t)

        tree['node_to_children'][s].append(t)
        tree['edge_to_label'][(s, t)] = y
        tree['edges'].append((s, t))

    return tree


def compute_pairwise_distance(tree):
    node_ids = tree['node_ids']
    node_TO_idx = {k: i for i, k in enumerate(node_ids)}

    n_a = len(node_ids)
    d = torch.zeros(n_a, n_a, dtype=torch.long)

    def helper(root):
        """ Compute pairwise distance between all descendants.
            Also includes distance to root.
        """
        children = tree['node_to_children'][root]

        if len(children) == 0:
            return [(root, 1)]

        # This contains [[(node_id, distance_to_root)]]
        descendants = []

        for x in children:
            x_descendants = helper(x)
            descendants.append(x_descendants)

        # Compute distance between children.
        for i, i_list in enumerate(descendants):
            for j, j_list in enumerate(descendants):
                if i == j:
                    continue

                for (i_node, i_root_dist) in i_list:
                    for (j_node, j_root_dist) in j_list:
                        i_node_id = node_TO_idx[i_node]
                        j_node_id = node_TO_idx[j_node]
                        assert d[i_node_id, j_node_id].item() == 0
                        d[i_node_id, j_node_id] = i_root_dist + j_root_dist


        # Compute distance from children to root.
        new_descendants = []
        j_node = root
        j_node_id = node_TO_idx[j_node]
        for i_list in descendants:
            for (i_node, i_root_dist) in i_list:
                i_node_id = node_TO_idx[i_node]

                assert d[i_node_id, j_node_id].item() == 0
                d[i_node_id, j_node_id] = i_root_dist

                assert d[j_node_id, i_node_id].item() == 0
                d[j_node_id, i_node_id] = i_root_dist

                new_descendants.append((i_node, i_root_dist + 1))

        new_descendants.append((root, 1))

        return new_descendants

    _ = helper(tree['root'])

    assert torch.all(d == d.transpose(0, 1)).item()

    return d
