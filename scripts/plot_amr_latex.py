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
from transition_amr_parser.amr import JAMR_CorpusReader
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
    parser.add_argument(
        "--sortby-recall",
        help="Sort examples by recall, only if gold-amr is provided.",
        action='store_true'
    )
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(1, 1e8)
    if args.out_tex is None:
        args.out_tex = args.in_amr + '.tex'
    return args


def read_amr(in_amr, unicode_fixes=False):

    corpus = JAMR_CorpusReader()
    corpus.load_amrs(in_amr)

    if unicode_fixes:

        # Replacement rules for unicode chartacters
        replacement_rules = {
            'ˈtʃærɪti': 'charity',
            '\x96': '_',
            '⊙': 'O'
        }

        # FIXME: normalization shold be more robust. Right now use the tokens
        # of the amr inside the oracle. This is why we need to normalize them.
        for idx, amr in enumerate(corpus.amrs):
            new_tokens = []
            for token in amr.tokens:
                forbidden = [x for x in replacement_rules.keys() if x in token]
                if forbidden:
                    token = token.replace(
                        forbidden[0],
                        replacement_rules[forbidden[0]]
                     )
                new_tokens.append(token)
            amr.tokens = new_tokens

    return corpus


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

        total, correct = 0, 0
        for node_id, span in gold.alignments.items():
            total += 1

            if node_id not in pred.alignments:
                continue

            if pred.alignments[node_id][0] in span:
                correct += 1

        recall = correct / total if total > 0 else 1
        return recall

    if args.gold_amr is not None:
        gold_corpus = read_amr(args.gold_amr).amrs
        if args.sortby_recall:
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
        
        latex_str = get_tikz_latex(amr, amr.tokens, amr.nodes, amr.edges, alignments)

        try:
            output_str = ''

            output_str += "\n\\resizebox{\\columnwidth}{!}{%\n"
            output_str += "\n\\begin{footnotesize}\n"
            output_str += latex_str
            output_str += "\n\end{footnotesize}\n"
            output_str += "\n}\n"
            output_str += 'pred\n'
            output_str += 'index={}\n'.format(index)
            output_str += 'tokens={}\n'.format(' '.join(amr.tokens))
            output_str += '%' + '\n'

            if args.gold_amr is not None:
                amr = gold_corpus[index]
                alignments = {k: v[0]-1 for k, v in amr.alignments.items()}

                latex_str = get_tikz_latex(amr, amr.tokens, amr.nodes, amr.edges, alignments)
                output_str += "\n\\resizebox{\\columnwidth}{!}{%\n"
                output_str += "\n\\begin{footnotesize}\n"
                output_str += latex_str
                output_str += "\n\end{footnotesize}\n"
                output_str += "\n}\n"

                if args.sortby_recall:
                    recall = sortkeys[index]
                    output_str += 'recall = {:.3f}\n'.format(recall)

                output_str += '%' + 'gold\n'
                output_str += '%' + 'index={}\n'.format(index)
                output_str += '%' + 'tokens={}\n'.format(' '.join(amr.tokens))
                output_str += '%' + '\n'

        except:
            print('WARNING: Skipping.')
            continue

        fdraw.write(output_str)

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
