import collections

import numpy as np

from ibm_neural_aligner.amr_utils import convert_amr_to_tree, compute_pairwise_distance, get_node_ids


def fertility_proxy(amr, ignore_nodes=('country', '-', 'and', 'person', 'name')):
    """ Measures the average number of aligned words per sentence.

        Lower indicates higher fertility.
    """
    alignments = amr.alignments.copy()

    for k in list(alignments.keys()):
        if ignore_nodes is not None and amr.nodes[k] in ignore_nodes:
            del alignments[k]

    return len(set([v[0] for k, v in alignments.items()]))


def distortion_proxy(amr, pairwise_dist=None):
    """ Measures the difference between implied and actual distance.

        Lower indicates lower distortion.
    """
    if len(amr.nodes) == 1 or len(amr.alignments) == 1:
        return 0, []

    if pairwise_dist is None:
        tree = convert_amr_to_tree(amr)
        pairwise_dist = compute_pairwise_distance(tree)
    node_ids = get_node_ids(amr)

    c = collections.defaultdict(list)

    for i in range(len(node_ids)):
        for j in range(len(node_ids)):
            if i <= j:
                continue
            node1, node2 = node_ids[i], node_ids[j]
            if node1 not in amr.alignments or node2 not in amr.alignments:
                continue
            pos1, pos2 = amr.alignments[node1][0], amr.alignments[node2][0]
            c['i'].append(i)
            c['j'].append(j)
            c['pos1'].append(pos1)
            c['pos2'].append(pos2)

    actual_distance = np.abs(np.array(c['pos1']) - np.array(c['pos2']))
    implied_distance = pairwise_dist[c['i'], c['j']].numpy()
    proxy = np.power(np.clip(actual_distance - implied_distance, 0, np.inf), 2)

    return proxy.mean(), proxy
