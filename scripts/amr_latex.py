#!/usr/bin/python
import sys
import re
import string
import os

def replace_symbols(line):

    line = line.replace("\\","")
    line = line.replace("$","\$")
    line = line.replace("#","*")
    line = line.replace("&","and")
    line = line.replace("%","\%")
    line = line.replace("_","-")
    line = line.replace("^","hat")
    line = line.replace("(","LBR")
    line = line.replace(")","RBR")
    line = line.replace("{","LBR")
    line = line.replace("}","RBR")

    return line

def get_tikz_latex(tokens, nodes, edges, alignments):

    for i in range(len(tokens)):
        tokens[i] = replace_symbols(tokens[i])
    for node in nodes:
        nodes[node]  = replace_symbols(nodes[node])

    latex_str = ""
    
    latex_str += "\\begin{center}\n\\begin{tikzpicture}[scale=1.5]\n"
    for i in range(0,len(tokens)):
        word = tokens[i]
        latex_str += "\\draw(" + str(float(i)*0.8) + ",0) node {" + word[0:10] + "};\n"

    children = {}
    for node in nodes:
        children[node] = []
        for edge in edges:
            if edge[0] == node:
                children[node].append(edge[2])

    node_keys = nodes.keys()
    levels = {}
    for node in nodes:
        lvl = node.count(".")
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append(node)
    max_lvl = max(levels.keys())
        
    node_names = {}
    for lvl in levels:
        y = 0.5 + (max_lvl - lvl) * 1.5
        for node in levels[lvl]:
            x=-0.8
            if node in alignments:
                x = float(alignments[node])*0.8
            node_names[node] = node.replace(".","_")
            latex_str += "\\node [draw,rounded corners] (" + str(node_names[node]) + ") at (" + str(x) + "," + str(y) + ") {" + nodes[node] + "};\n"
    '''
    plotted = []
    previous_plotted = []
    level = 0
    while len(plotted) != len(nodes):
        for node in nodes:
            if node not in plotted and (len(children[node]) == 0 or all(child in previous_plotted for child in children[node])):
                #plot this nodes
                x=-0.8
                if node in alignments:
                    x = float(alignments[node])*0.8
                y = 0.5 + level * 1.5
                node_names[node] = node.replace(".","_")
                latex_str += "\\node [draw,rounded corners] (" + str(node_names[node]) + ") at (" + str(x) + "," + str(y) + ") {" + nodes[node] + "};\n"
                plotted.append(node)
        level += 1
        previous_plotted = plotted
    '''
    
    for edge in edges:
        latex_str += "\\draw [-latex,thick] (" + node_names[edge[0]] + ") -- node {\\footnotesize " + replace_symbols(edge[1]) + "} (" + node_names[edge[2]] + ");\n"
        
    latex_str += "\\end{tikzpicture}\n\\end{center}\n"

    return latex_str
