# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
# https://matplotlib.org/3.1.1/tutorials/text/annotations.html#placing-artist-at-the-anchored-location-of-the-axes
# FIXME: Separate rendering and node position calculation
# FIXME: Variable names messy
import argparse
import json
import random
import re
import numpy as np
from random import shuffle
from transition_amr_parser.io import read_amr
from amr_latex import get_tikz_latex

def argument_parser():

    parser = argparse.ArgumentParser(description='AMR alignment plotter')
    # Single input parameters
    parser.add_argument(
        "--max",
        help=".",
        type=int,
        default=100
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        type=int,
        default=None
    )
    parser.add_argument(
        "--in-amr",
        help="AMR 2.0+ annotation file to be splitted",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-tex",
        help="output",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--gold-amr",
        help="Compare with gold to find interesting examples.",
        type=str,
        default=None,
        required=False
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
    if args.seed is None:
        args.seed = random.randint(1, 1e8)
    if args.out_tex is None:
        args.out_tex = args.in_amr + '.tex'
    return args

def main(args):
    print(json.dumps(args.__dict__))

    random.seed(args.seed)

    corpus = read_amr(args.in_amr).amrs
    print(f'Read {args.in_amr}')
    num_amrs = len(corpus)

    # Sort or shuffle.
    if args.indices:
        indices = list(map(int, args.indices))
    else:
        indices = list(range(num_amrs))
        shuffle(indices)

    # Optionally compare with gold.
    def compare_align(gold, pred):
        def get_pairs(x):
            node_ids = x.nodes.keys()
            alignments = [x.alignments[node_id][0] - 1 for node_id in node_ids if node_id in x.alignments]
            pairs = [(node_id, a) for node_id, a in zip(node_ids, alignments)]
            return set(pairs)
        g = get_pairs(gold)
        p = get_pairs(pred)

        total = len(g)
        found = len(set.intersection(g, p))
        recall = found / total if total > 0 else 1
        return recall

    if args.gold_amr is not None:
        gold_corpus = read_amr(args.gold_amr).amrs
        sortkeys = [compare_align(g, p) for g, p in zip(gold_corpus, corpus)]
        indices = np.argsort(sortkeys)

    fname = args.out_tex if args.out_tex.endswith(".tex") else args.out_tex + ".tex"
    fdraw = open(fname,'w')
    fdraw.write("\\documentclass[landscape,letterpaper]{article}\n\\usepackage[left=5pt,top=5pt,right=5pt]{geometry}\n\\usepackage{tikz}\n\\begin{document}\n")

    def token_filter(tokens):
        # Attempts to remove less interesting tokens such as single characters and symbols, which often occur in URLs.
        return [tok for tok in tokens if tok.isalpha() and len(tok) > 1]
        
    # get one sample
    found = 0
    for index in indices:

        amr = corpus[index]
        if len(amr.tokens) < 15:
            continue

        if len(token_filter(amr.tokens)) < 10:
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
            and len(set(token_filter(amr.tokens))) == len(token_filter(amr.tokens))
        ):
            continue

        print('\n'.join([
            x
            for x in amr.toJAMRString().split('\n')
            if not x.startswith('# ::edge')
        ]))
        print(index)

        # plot
        alignments = {k: v[0]-1 for k, v in amr.alignments.items()}
        
        latex_str = get_tikz_latex(amr.tokens, amr.nodes, amr.edges, alignments)
        fdraw.write("\n\\begin{footnotesize}\n")
        fdraw.write(latex_str)
        fdraw.write("\n\end{footnotesize}\n")
        fdraw.write('%' + 'pred\n')
        fdraw.write('%' + 'index={}\n'.format(index))
        fdraw.write('%' + 'tokens={}\n'.format(' '.join(amr.tokens)))
        fdraw.write('%' + '\n')

        if args.gold_amr is not None:
            amr = gold_corpus[index]
            alignments = {k: v[0]-1 for k, v in amr.alignments.items()}

            latex_str = get_tikz_latex(amr.tokens, amr.nodes, amr.edges, alignments)
            fdraw.write("\n\\begin{footnotesize}\n")
            fdraw.write(latex_str)
            fdraw.write("\n\end{footnotesize}\n")

            recall = sortkeys[index]

            fdraw.write('recall = {:.3f}\n'.format(recall))

            fdraw.write('%' + 'gold\n')
            fdraw.write('%' + 'index={}\n'.format(index))
            fdraw.write('%' + 'tokens={}\n'.format(' '.join(amr.tokens)))
            fdraw.write('%' + '\n')

        found += 1
        
        if found >= args.max:
            break
        else:
            continue
        
        response = input('Quit [N/y]?')
        if response == 'y':
            break

    fdraw.write("\n\end{document}")
        
if __name__ == '__main__':

    # Argument handling
    main(argument_parser())
