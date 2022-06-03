import re
import subprocess
from ipdb import set_trace
from transition_amr_parser.amr import AMR
from transition_amr_parser.amr_constituents import (
    NodeDepth,
    get_reentrant_edges
)


def replace_symbols(line):

    line = line.replace("\\", "")
    line = line.replace("$", r"\$")
    line = line.replace("#", "*")
    line = line.replace("&", "and")
    line = line.replace("%", r"\%")
    line = line.replace("_", "-")
    line = line.replace("^", "hat")
    # line = line.replace("(", "LBR")
    # line = line.replace(")", "RBR")
    line = line.replace("{", r"\{")
    line = line.replace("}", r"\}")

    return line


def pdflatex(tex_file):
    stdout, stderr = subprocess.Popen(
        f'pdflatex {tex_file}',
        stdout=subprocess.PIPE,
        shell=True
    ).communicate()
    return re.sub('tex$', 'pdf', tex_file)


def save_graphs_to_tex(tex_file, amr_str, plot_cmd=None):

    with open(tex_file, 'w') as fid:
        fid.write(document_template(amr_str))

    if plot_cmd:
        pdf_file = pdflatex(tex_file)
        stdout, stderr = subprocess.Popen(
            f'{plot_cmd} {pdf_file}', stdout=subprocess.PIPE, shell=True
        ).communicate()


def save_amr_to_tex(tex_file, amr, plot_cmd=None, color_by_id={},
                    color_by_id_pair={}, scale=1):

    amr_str = get_tikz_latex(
        amr,
        color_by_id=color_by_id, color_by_id_pair=color_by_id_pair,
        scale=scale
    )
    save_graphs_to_tex(tex_file, amr_str, plot_cmd=plot_cmd)


def document_template(body_tex):

    latex_str = ''
    latex_str += "\\documentclass[landscape,letterpaper]{article}\n"
    latex_str += '\\usepackage[left=5pt,top=5pt,right=5pt]{geometry}\n'
    latex_str += '\\usepackage{tikz}\n'
    latex_str += '\\begin{document}\n\n'
    latex_str += '\\begin{footnotesize}\n'
    latex_str += body_tex
    latex_str += "\n\\end{footnotesize}\\end{document}"

    return latex_str


def picture_template(graph_latex_str, scale=1):
    latex_str = ''
    latex_str += '\\begin{center}\n'
    latex_str += f'\\begin{{tikzpicture}}[scale={scale}]\n'
    latex_str += graph_latex_str
    latex_str += '\\end{tikzpicture}\n'
    latex_str += '\\end{center}\n'
    return latex_str


def format(tokens, nodes, edges, alignments):

    # filter out invalid symbols
    for i in range(len(tokens)):
        # truncate at 10 char
        tokens[i] = replace_symbols(tokens[i])[:10]
    for node in nodes:
        nodes[node] = replace_symbols(nodes[node])

    # edges
    edges = [(s, replace_symbols(e), t) for (s, e, t) in edges]

    # use first alignment
    alignments = {k: v[0] for k, v in alignments.items()}

    # identify wikis
    wiki_ids = {}
    for (src, label, trg) in edges:
        if label == ':wiki':
            wiki_ids[trg] = src

    return tokens, nodes, edges, alignments, wiki_ids


def get_tikz_latex(
    amr: AMR,
    color_by_id={},
    color_by_id_pair={},
    scale=1,
    x_warp=1.0,
    y_warp=1.0
):

    # get re-entrant edges
    reentrant_edges = get_reentrant_edges(amr, alignment_sort=True)

    # affine transformations
    y_bias = 0.5 * y_warp

    # GRID precomputed here
    # get the depth level of each node
    node_depth = NodeDepth()
    node_depth.reset(amr)
    node_depth.trasverse()
    # levels, max_j = get_depths(amr.nodes, amr.edges)
    levels = node_depth.grid
    max_j = max(levels)
    # precompute lengths
    pos_y = [y_bias + (max_j - j) * y_warp for j in sorted(levels.keys())]
    # precompute horizontal positions
    pos_x = [float(i) * x_warp for i in range(len(amr.tokens))]

    # format AMR components and extract wiki
    amr.tokens, amr.nodes, amr.edges, amr.alignments, wiki_ids = \
        format(amr.tokens, amr.nodes, amr.edges, amr.alignments)

    # generate tikz latex string
    # tokens
    latex_str = "% tokens \n"
    for i in range(len(amr.tokens)):
        y = y_bias - 0.5 * y_warp
        latex_str += f'\\draw({pos_x[i]:.2f},{y}) node {{{amr.tokens[i]}}};\n'

    # nodes
    latex_str += "% nodes\n"
    pos_by_id = {}
    # loop over levels
    for j in range(len(pos_y)):

        if j not in levels:
            continue

        y = pos_y[j]
        # for node at that level
        for node in levels[j]:

            node_options = []

            # color
            if node in color_by_id:
                node_options.append('thick')
                node_options.append(f'draw={color_by_id[node]}')
                label_str = f'{{\\color{{{color_by_id[node]}}}'
                label_str += ' {amr.nodes[node]}}}'
            else:
                node_options.append('draw')
                label_str = f'{amr.nodes[node]}'

            node_options.append('rounded corners')

            # default position
            if node in wiki_ids:
                # wiki are located as same height as NER label and right most
                y = pos_by_id[wiki_ids[node]][1]
                x = float(len(amr.tokens)) * x_warp
                lvl_shift = j
                while (x, y) in (
                    pos_by_id.values() or lvl_shift >= len(pos_y)
                ):
                    lvl_shift += 1
                    y = pos_y[lvl_shift]
                node_options.append('anchor=west')
            elif node in amr.alignments:
                # alignment position
                x = pos_x[amr.alignments[node]]
            else:
                # default no alignment
                x = -x_warp

            # node
            option_str = ','.join(node_options)
            latex_str += f'\\node [{option_str}] '
            latex_str += f'({node}) at ({x},{y}) {{{label_str}}};\n'

            # store position for later
            pos_by_id[node] = (x, y)

    # ensure all nodes printed
    if set(pos_by_id.keys()) != set(amr.nodes.keys()):
        set_trace(context=30)

    assert set(pos_by_id.keys()) == set(amr.nodes.keys()), \
        "Some nodes could not be printed"

    # draw edge
    latex_str += "% edges\n"
    for (src, label, trg) in amr.edges:
        # options
        if (src, label, trg) in reentrant_edges:
            draw_options = ['-latex', 'thick', 'right', 'dashed']
        else:
            draw_options = ['-latex', 'thick', 'right']
        # color
        if (src, trg) in color_by_id_pair:
            draw_options.append(f'{color_by_id_pair[(src, trg)]}')
        # wiki edge orientation
        if label == ':wiki':
            draw_options.append('above')
        else:
            draw_options.append('right')
        option_str = ','.join(draw_options)

        # edge
        latex_str += f'\\draw [{option_str}] ({src}) -- node '
        latex_str += f'{{\\footnotesize {label}}} ({trg});\n'

    return picture_template(latex_str, scale=scale)
