import re
import sys
from collections import Counter

from amr import JAMR_CorpusReader


def main():
    cr = JAMR_CorpusReader()
    cr.load_amrs(sys.argv[1], verbose=False)

    special_alignments = {}

    for amr in cr.amrs:
        for node_id in amr.alignments:
            aligned_token_ids = amr.alignments[node_id]
            aligned_node_ids = amr.alignmentsToken2Node(aligned_token_ids[0])
            aligned_node_ids = [id for id in aligned_node_ids if '"' not in amr.nodes[id]]
            if len(aligned_node_ids) <= 1:
                continue

            subgraph = amr.findSubGraph(aligned_node_ids)
            # normalize named entities
            if len(subgraph.edges) == 1 and subgraph.edges[0][1] == ':name':
                subgraph.nodes[subgraph.root] = '[entity]'
            # normalize numbers
            for n in subgraph.nodes:
                if re.match('[0-9]+', subgraph.nodes[n]):
                    subgraph.nodes[n] = '[NUM]'
                if subgraph.nodes[n].endswith('quantity'):
                    subgraph.nodes[n] = '[quantity]'
                # if subgraph.nodes[n].endswith('entity'):
                #     subgraph.nodes[n] = '[value]'
            aligned_subgraph = str(subgraph)

            aligned_tokens = ' '.join(amr.tokens[x] for x in aligned_token_ids if x < len(amr.tokens))

            if aligned_subgraph not in special_alignments:
                special_alignments[aligned_subgraph] = Counter()
            special_alignments[aligned_subgraph][aligned_tokens] += 1

    for special in sorted(special_alignments, key=lambda x: sum(special_alignments[x].values()), reverse=True):
        print(special, sum(special_alignments[special].values()))
        print(special_alignments[special].most_common(10))
        print('\n')


if __name__ == '__main__':
    main()
