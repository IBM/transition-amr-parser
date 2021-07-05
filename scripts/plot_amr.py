# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
# https://matplotlib.org/3.1.1/tutorials/text/annotations.html#placing-artist-at-the-anchored-location-of-the-axes
# FIXME: Separate rendering and node position calculation
# FIXME: Variable names messy
from random import shuffle
import argparse
from transition_amr_parser.io import read_amr
from transition_amr_parser.plots import plot_graph, convert_format


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR alignment plotter')
    # Single input parameters
    parser.add_argument(
        "--in-amr",
        help="AMR 2.0+ annotation file to be splitted",
        type=str,
        required=True
    )
    parser.add_argument(
        "--indices", nargs='+',
        help="Position on the AMR file of sentences to plot"
    )
    parser.add_argument(
        "--has-nodes", nargs='+',
        help="filter for AMRs that have those nodes"
    )
    parser.add_argument(
        "--has-repeated-nodes",
        help="filter for AMRs that have more than one node of same name",
        action='store_true'
    )
    parser.add_argument(
        "--has-repeated-tokens",
        help="filter for AMRs that have more than one node of same name",
        action='store_true'
    )
    parser.add_argument(
        "--has-edges", nargs='+',
        help="filter for AMRs that have those nodes"
    )
    args = parser.parse_args()
    return args


def get_example():

    # fake graph
    tokens = ["The", "boy", "wants", "to", "go", "to", "New", "York"]
    #          0        1         2       3       4       5      6
    nodes = ['boy', 'want-01', 'go-02', 'city', 'name', 'New', 'York']
    edges = [
        (1, 'ARG0', 0),
        (1, 'ARG1', 2),
        (2, 'ARG0', 0),
        (2, 'ARG4', 3),
        (3, 'name', 4),
        (4, 'op1', 5),
        (4, 'op2', 6)
    ]
    alignments = [[], [0], [1], [], [2], [], [5, 4, 3], [6, 4, 3]]
    return tokens, nodes, edges, alignments


def main(args):

    corpus = read_amr(args.in_amr).amrs
    print(f'Read {args.in_amr}')
    num_amrs = len(corpus)
    if args.indices:
        indices = list(map(int, args.indices))
    else:
        indices = list(range(num_amrs))
        shuffle(indices)

    # get one sample
    for index in indices:

        amr = corpus[index]

        if 7 > len(amr.tokens) > 10:
            continue

        # Get tokens aligned to nodes
        aligned_tokens = [
            amr.tokens[i-1]
            for indices in amr.alignments.values()
            for i in indices
        ]

        # skip amr not meeting criteria
        if (
            args.has_nodes
            and not set(args.has_nodes) <= set(amr.nodes.values())
        ) or (
            args.has_edges
            and not set(args.has_edges) <= set([x[1][1:] for x in amr.edges])
        ) or (
            args.has_repeated_nodes
            and len(set(amr.nodes.values())) == len(amr.nodes.values())
        ) or (
            args.has_repeated_tokens
            and len(set(aligned_tokens)) == len(aligned_tokens)
        ):
            continue

        # convert IBM AMR format to the one used here
        # tokens, nodes, edges, alignments = convert_format(amr)

        print('\n'.join([
            x
            for x in amr.toJAMRString().split('\n')
            if not x.startswith('# ::edge')
        ]))
        print(index)

        # plot
        alignments = {k: v[0]-1 for k, v in amr.alignments.items()}
        plot_graph(amr.tokens, amr.nodes, amr.edges, alignments)

        response = input('Quit [N/y]?')
        if response == 'y':
            break


if __name__ == '__main__':

    # Argument handling
    main(argument_parser())
