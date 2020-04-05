import sys
from transition_amr_parser.utils import print_log


class AMR:

    def __init__(self, tokens=None, root='', nodes=None, edges=None, alignments=None, score=0.0):

        if alignments is None:
            alignments = {}
        if edges is None:
            edges = []
        if nodes is None:
            nodes = {}
        if tokens is None:
            tokens = []
        self.tokens = tokens
        self.root = root
        self.nodes = nodes
        self.edges = edges
        self.alignments = alignments
        self.score = score

        self.token2node_memo = {}

    def __str__(self):
        output = ''
        # tokens
        output += '# ::tok ' + (' '.join(self.tokens)) + '\n'
        # score
        if self.score:
            output += f'# ::scr\t{self.score}\n'
        # nodes
        for n in self.nodes:
            alignment = ''
            if n in self.alignments and self.alignments[n]:
                if type(self.alignments[n]) == int:
                    alignment = f'\t{self.alignments[n]-1}-{self.alignments[n]}'
                else:
                    alignments_in_order = sorted(list(self.alignments[n]))
                    alignment = f'\t{alignments_in_order[0]-1}-{alignments_in_order[-1]}'
            output += f'# ::node\t{n}\t{self.nodes[n] if n in self.nodes else "None"}' + alignment + '\n'
        # root
        root = self.root
        alignment = ''
        # if root in self.alignments and self.alignments[root]:
        #     if type(self.alignments[root]) == int:
        #         alignment = f'\t{self.alignments[root]-1}-{self.alignments[root]}'
        #     else:
        #         alignments_in_order = sorted(list(self.alignments[root]))
        #         alignment = f'\t{alignments_in_order[0]-1}-{alignments_in_order[-1]}'
        if self.root:
            output += f'# ::root\t{root}\t{self.nodes[root] if root in self.nodes else "None"}' + alignment + '\n'
        # edges
        for s, r, t in self.edges:
            r = r.replace(':', '')
            output += f'# ::edge\t{self.nodes[s] if s in self.nodes else "None"}\t{r}\t{self.nodes[t] if t in self.nodes else "None"}\t{s}\t{t}\t\n'
        return output

    def alignmentsToken2Node(self, token_id):
        if token_id not in self.token2node_memo:
            self.token2node_memo[token_id] = sorted([node_id for node_id in self.alignments if token_id in self.alignments[node_id]])
        return self.token2node_memo[token_id]

    def copy(self):
        return AMR(self.tokens.copy(), self.root, self.nodes.copy(), self.edges.copy(), self.alignments.copy(), self.score)

    """
    Outputs all edges such that source and target are in node_ids
    """

    def findSubGraph(self, node_ids):
        if not node_ids:
            return AMR()
        potential_root = node_ids.copy()
        sg_edges = []
        for x, r, y in self.edges:
            if x in node_ids and y in node_ids:
                sg_edges.append((x, r, y))
                if y in potential_root:
                    potential_root.remove(y)
        root = potential_root[0] if len(potential_root) > 0 else node_ids[0]
        return AMR(root=root,
                   edges=sg_edges,
                   nodes={n: self.nodes[n] for n in node_ids})

    def toJAMRString(self, allow_incomplete=False):
        output = str(self)

        # amr string
        amr_string = f'[[{self.root}]]'
        new_ids = {}
        for n in self.nodes:
            new_id = self.nodes[n][0] if self.nodes[n] else 'x'
            if new_id.isalpha() and new_id.islower():
                if new_id in new_ids.values():
                    j = 2
                    while f'{new_id}{j}' in new_ids.values():
                        j += 1
                    new_id = f'{new_id}{j}'
            else:
                j = 0
                while f'x{j}' in new_ids.values():
                    j += 1
                new_id = f'x{j}'
            new_ids[n] = new_id
        depth = 1
        nodes = {self.root}
        completed = set()
        while '[[' in amr_string:
            tab = '      '*depth
            for n in nodes.copy():
                id = new_ids[n] if n in new_ids else 'r91'
                concept = self.nodes[n] if n in new_ids and self.nodes[n] else 'None'
                edges = sorted([e for e in self.edges if e[0] == n], key=lambda x: x[1])
                targets = set(t for s, r, t in edges)
                edges = [f'{r} [[{t}]]' for s, r, t in edges]
                children = f'\n{tab}'.join(edges)
                if children:
                    children = f'\n{tab}'+children
                if n not in completed:
                    if (concept[0].isalpha() and concept not in ['imperative', 'expressive', 'interrogative']) or targets:
                        amr_string = amr_string.replace(f'[[{n}]]', f'({id} / {concept}{children})', 1)
                    else:
                        amr_string = amr_string.replace(f'[[{n}]]', f'{concept}')
                    completed.add(n)
                amr_string = amr_string.replace(f'[[{n}]]', f'{id}')
                nodes.remove(n)
                nodes.update(targets)
            depth += 1

        if allow_incomplete:
            pass

        else:
            if len(completed) < len(self.nodes):
                raise Exception("Tried to print an uncompleted AMR")
                print_log('amr', 'Failed to print AMR, ' + str(len(completed)) + ' of ' + str(len(self.nodes)) + ' nodes printed:\n ' + amr_string)
            if amr_string.startswith('"') or amr_string[0].isdigit() or amr_string[0] == '-':
                amr_string = '(x / '+amr_string+')'
            if not amr_string.startswith('('):
                amr_string = '('+amr_string+')'
            if len(self.nodes) == 0:
                amr_string = '(a / amr-empty)'

            output += amr_string + '\n\n'

        return output


class JAMR_CorpusReader:

    special_tokens = []

    def __init__(self):
        self.amrs = []
        self.amrs_dev = []

        # index dictionaries
        self.nodes2Ints = {}
        self.words2Ints = {}
        self.chars2Ints = {}
        self.labels2Ints = {}

        print_log('amr', 'Starts reading data')

    """
    Reads AMR Graphs file in JAMR format. If Training==true, it is reading
    training data set and it will affect the dictionaries.

    JAMR format is ...
    # ::scr score
    # ::tok tokens...
    # ::node node_id node alignments
    # ::root root_id root
    # ::edge src label trg src_id trg_id
    amr graph
    """

    def load_amrs(self, amr_file_name, training=True, verbose=False):

        amrs = self.amrs if training else self.amrs_dev

        amrs.append(AMR())

        fp = open(amr_file_name, encoding='utf8')
        for line in fp:
            # empty line, prepare to read next amr in dataset
            if len(line.strip()) == 0:
                if verbose:
                    print(amrs[-1])
                amrs.append(AMR())
            # amr tokens
            elif line.startswith("# ::tok"):
                tokens = line[len('# ::tok '):]
                tokens = tokens.split()
                amrs[-1].tokens.extend(tokens)
                for tok in tokens:
                    # TODO: update dictionaries after entire AMR is read
                    if training:
                        self.words2Ints.setdefault(tok, len(self.words2Ints))
                        for char in tok:
                            self.chars2Ints.setdefault(char, len(self.chars2Ints))
            # amr score
            elif line.startswith("# ::scr"):
                score = line.strip()[len('# ::scr '):]
                score = float(score)
                amrs[-1].score = score
            # an amr node
            elif line.startswith("# ::node"):
                node_id = ''
                for col, tab in enumerate(line.split("\t")):
                    # node id
                    if col == 1:
                        node_id = tab.strip()
                        # node label
                    elif col == 2:
                        node = tab.strip()
                        amrs[-1].nodes[node_id] = node
                        # TODO: update dictionaries after entire AMR is read
                        if training:
                            self.nodes2Ints.setdefault(node, len(self.nodes2Ints))
                    # alignment
                    elif col == 3:
                        if '-' not in tab:
                            continue
                        start_end = tab.strip().split("-")
                        start = int(start_end[0])  # inclusive
                        end = int(start_end[1])  # exclusive
                        word_idxs = list(range(start+1, end+1))  # off by one (we start at index 1)
                        amrs[-1].alignments[node_id] = word_idxs

            # an amr edge
            elif line.startswith("# ::edge"):
                edge = ['', '', '']
                in_quotes = False
                quote_offset = 0
                for col, tab in enumerate(line.split("\t")):
                    if tab.startswith('"'):
                        in_quotes = True
                    if tab.endswith('"'):
                        in_quotes = False
                    # edge label
                    if col == 2 + (quote_offset):
                        edge[1] = ':'+tab.strip()
                        # TODO: update dictionaries after entire AMR is read
                        if training:
                            self.labels2Ints.setdefault(tab, len(self.labels2Ints))
                    # edge source id
                    elif col == 4 + (quote_offset):
                        edge[0] = tab.strip()
                    # edge target id
                    elif col == 5 + (quote_offset):
                        edge[2] = tab.strip()
                    if in_quotes:
                        quote_offset += 1
                amrs[-1].edges.append(tuple(edge))
            # amr root
            elif line.startswith("# ::root"):
                splinetabs = line.split("\t")
                root = splinetabs[1]
                root = root.strip()
                amrs[-1].root = root

        if len(amrs[-1].nodes) == 0:
            amrs.pop()
        print_log('amr', "Training Data" if training else "Dev Data")
        if training:
            print_log('amr', "Number of labels: " + str(len(self.labels2Ints)))
            print_log('amr', "Number of nodes: " + str(len(self.nodes2Ints)))
            print_log('amr', "Number of words: " + str(len(self.words2Ints)))
        print_log('amr', "Number of sentences: " + str(len(amrs)))


def main():
    file = sys.argv[1] if len(sys.argv) > 1 else "our_aln_2016.txt"

    cr = JAMR_CorpusReader()
    cr.load_amrs(file, verbose=True)


if __name__ == '__main__':
    main()
