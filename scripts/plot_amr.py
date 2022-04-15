# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
# https://matplotlib.org/3.1.1/tutorials/text/annotations.html#placing-artist-at-the-anchored-location-of-the-axes
# FIXME: Separate rendering and node position calculation
# FIXME: Variable names messy
from random import shuffle
import argparse
from transition_amr_parser.io import read_amr, read_endpoint_amr
from transition_amr_parser.amr_latex import (
    get_tikz_latex,
    save_graphs_to_tex,
)
from ipdb import set_trace


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
        "--shuffle",
        help="randomize input AMRs",
        action='store_true'
    )
    parser.add_argument(
        "--jamr",
        help="Read from JAMR annotations",
        action='store_true'
    )
    parser.add_argument(
        "--out-tex",
        help="output",
        type=str,
        required=True
    )
    # latex / tikz variables
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--x-warp", type=float, default=1.0)
    parser.add_argument("--y-warp", type=float, default=1.0)
    #
    parser.add_argument(
        "--max-graphs",
        help="Will stop after plotting this amount",
        default=100,
        type=int,
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


def fix_ner_alignments(amr):

    # fix alignments
    for (src, edge, trg) in amr.edges:
        if edge == ':name' and amr.nodes[trg] == 'name':
            ops = sorted(amr.children(trg), key=lambda x: [1])[::-1]
            if (
                len(amr.alignments[trg]) > 1
                and len(amr.alignments[trg]) == len(ops)
            ):
                for idx, (nid, _) in enumerate(ops):
                    amr.alignments[nid] = [amr.alignments[trg][idx]]

    return amr


def skip_amr(amr, args):
    return (
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
        and len(set(amr.tokens)) == len(amr.tokens)
    )


def main(args):

    # argument handling
    if args.in_amr.endswith('json'):
        # assumed JSON from endpoint
        amrs = read_endpoint_amr(args.in_amr)
    else:
        amrs = read_amr(args.in_amr, ibm_format=args.jamr)

    print(f'Read {args.in_amr}')
    num_amrs = len(amrs)
    if args.indices:
        indices = list(map(int, args.indices))
    else:
        indices = list(range(num_amrs))
    # write into file
    tex_file = args.out_tex
    if args.shuffle:
        shuffle(indices)

    # get one sample
    amr_strs = []
    for index in indices:

        amr = amrs[index]

        # Fix NER
        amr = fix_ner_alignments(amr)

        # Remove ROOT
        if amr.tokens[-1] == '<ROOT>':
            amr.tokens = amr.tokens[:-1]

        if len(amr_strs) >= args.max_graphs:
            # too many graphs
            break

        # skip amr not meeting criteria
        if skip_amr(amr, args) or amr.edges == []:
            continue

        src,  _, trg = amr.edges[0]

        # get latex string
        amr_str = get_tikz_latex(
            amr,
            # color_by_id={'a': 'red'},
            # color_by_id_pair={(src, trg): 'red'},
            scale=args.scale,
            x_warp=args.x_warp,
            y_warp=args.y_warp
        )

        # plot
        amr_strs.append(amr_str)

        # open on the fly
        save_graphs_to_tex(tex_file, amr_str, plot_cmd='open')

        response = input('Quit [N/y]?')
        if response == 'y':
            break

    # write all graphs to a single tex
    print(f'Wrote {len(amr_strs)} amrs into {tex_file}')
    save_graphs_to_tex(tex_file, '\n'.join(amr_strs))


if __name__ == '__main__':
    # argument handling
    main(argument_parser())
