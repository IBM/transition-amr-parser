from collections import defaultdict
from ipdb import set_trace
try:
    import matplotlib.pyplot as plt
except ImportError:
    # be silent here as this is optional
    pass


def convert_format(amr):

    # convert to this format
    tokens = amr.tokens
    nodes = list(amr.nodes.values())
    label2index = {name: i for i, name in enumerate(amr.nodes.keys())}
    edges = [
        [label2index[edge[0]], edge[1][1:], label2index[edge[2]]]
        for edge in amr.edges
    ]
    alignments = [[] for _ in tokens]
    for node_id, token_pos in amr.alignments.items():
        for pos in token_pos:
            alignments[pos-1].append(label2index[node_id])

    return tokens, nodes, edges, alignments


def get_paths_to_root(surface_aligned_nodes, node_ids, edges):

    edge_by_child = defaultdict(list)
    for (parent, label, child) in edges:
        edge_by_child[child].append((label, parent))

    # Find all paths from surface linked nodes to the root, sort by length and
    # create as many levels as the maximum length
    # basic aligned graph data
    root_ids = [n for n in node_ids if n not in edge_by_child]
    paths_to_root = []
    for surface_node_id in surface_aligned_nodes:
        node_id = surface_node_id
        paths = [[node_id]]
        count = 0
        while count < 100:
            paths2 = []
            num_parents = 0
            for path in paths:
                for path_num, x in enumerate(edge_by_child[path[-1]]):
                    num_parents += 1
                    if x[1] in root_ids:
                        paths_to_root.append(path + [x[1]])
                    paths2.append(path + [x[1]])
            if num_parents == 0:
                break
            else:
                paths = paths2
            count += 1

    return paths_to_root, root_ids


def plot_graph(
        tokens, nodes, edges, aligned_token_by_node,
        mark_ids=None, plot_now=True, figsize=(5, 5), paper_height=200
    ):

    assert isinstance(tokens, list)
    assert isinstance(nodes, dict)
    assert isinstance(edges, list)
    assert isinstance(aligned_token_by_node, dict)

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    # determine positions of surface tokens in plot. All furtehr plotting
    # uses this as reference
    sentence_y = 5
    alignment_width_y = 2

    # localize all leaf nodes
    edge_by_parent = defaultdict(list)
    for (parent, label, child) in edges:
        edge_by_parent[parent].append((label, child))
    leaf_nodes = []
    for node_id in nodes:
        if node_id not in edge_by_parent:
            leaf_nodes.append(node_id)

    # get all paths from nodes aligned to surface symbols to the root
    paths_to_root, root_ids = get_paths_to_root(leaf_nodes, nodes.keys(), edges)
    if paths_to_root == []:
        return

    label_by_nodes = defaultdict(dict)
    for (parent, label, child) in edges:
        label_by_nodes[parent][child] = label

    # plot tokens
    token_x = 4
    token_plot_x = []
    for pos, token in enumerate(tokens):
        token_plot_x.append(token_x)
        ax.text(token_x, sentence_y, token, fontsize=12, ha='center',
                va='center')
        token_x += min(5, max(0.5 * len(token), 2))

    # determine position in plot of every node from its predecesor or token
    # alignments
    artist_by_nodeid = {}
    vertical_warp = 2
    position_cache = {}
    for path in sorted(paths_to_root, key=len, reverse=True):
        for depth, node_id in enumerate(path):
            if node_id not in aligned_token_by_node:
                # skip unaligned nodes
                continue
            elif node_id in position_cache:
                position = position_cache[node_id]
            else:
                alignments = aligned_token_by_node[node_id]
                if len(alignments) > 1:
                    pos_x = token_plot_x[alignments[0]]
                    pos_x += token_plot_x[alignments[-1]]
                    pos_x *= 0.5
                else:
                    pos_x = token_plot_x[alignments[0]]

                position = (
                    pos_x,
                    sentence_y + alignment_width_y + vertical_warp * depth
                )
                position_cache[node_id] = position
            # edge
            if depth > 0:

                prev_position = artist_by_nodeid[path[depth-1]][1]
                delta_x = prev_position[0] - position[0]
                delta_y = prev_position[1] - position[1]
                inc_y = 0.0
                inc_x = inc_y * delta_x / delta_y

                an1 = ax.annotate(
                    nodes[node_id],
                    xy=(
                        .85 * (delta_x + inc_x) + position[0],
                        .85 * (delta_y + inc_y) + position[1]
                    ),
                    va="center",
                    ha="center",
                    bbox=dict(
                        boxstyle="round",
                        fc="r" if mark_ids and node_id in mark_ids else 'w',
                        alpha=0.5 if mark_ids and node_id in mark_ids else 1.0
                    ),
                    #
                    xytext=position,
                    arrowprops=dict(arrowstyle="->")
                )

                # arc label
                ax.text(
                    .5 * (delta_x) + position[0],
                    .5 * (delta_y) + position[1],
                    label_by_nodes[node_id][path[depth-1]],
                    fontsize=10,
                    ha='left',
                    va='center'
                )

            else:
                an1 = ax.annotate(
                    nodes[node_id],
                    xy=position,
                    xycoords="data",
                    va="center",
                    ha="center",
                    bbox=dict(
                        boxstyle="round",
                        fc="r" if mark_ids and node_id in mark_ids else 'w',
                        alpha=0.5 if mark_ids and node_id in mark_ids else 1.0
                    )
                )
            artist_by_nodeid[node_id] = (an1, position)

    ax.set_xlim(0, token_plot_x[-1] + 5)
    paper_height = max([x[1] for x in position_cache.values()]) + 5
    ax.set_ylim(0, paper_height)

    plt.axis('off')
    plt.tight_layout()
    if plot_now:
        plt.show()
