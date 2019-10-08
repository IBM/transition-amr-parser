
import sys
import json as J
from collections import Counter
from amr import JAMR_CorpusReader

entity_rules_json = {}

verbose = True


def fix_alignments(amr):
    changes = 0
    for node_id in amr.nodes:
        if node_id not in amr.alignments:
            amr.alignments[node_id] = []
    # fix false positives
    for node_id in amr.nodes:
        token_ids = amr.alignments[node_id]
        if not token_ids:
            continue
        tokens = [amr.tokens[t - 1].lower() for t in token_ids if 0 <= t <= len(amr.tokens)]
        nodes = amr.alignmentsToken2Node(token_ids[0])

        if amr.nodes[node_id] == 'and':
            if len(nodes)>1 and not any(tok in ['and','&',';',','] for tok in tokens):
                amr.alignments[node_id] = []
                print('[removing alignment]', 'and', f'({" ".join(tokens)})')
                changes += 1
        elif amr.nodes[node_id] == 'i':
            if not any(tok in ['i', 'me', 'myself', 'my', 'mine','im','ill','i\'m','i\'ll'] for tok in tokens):
                amr.alignments[node_id] = []
                print('[removing alignment]', 'i', f'({" ".join(tokens)})')
                changes += 1
        elif amr.nodes[node_id] == 'you':
            if not any(tok.startswith('you') or tok.startswith('ya') or tok=='u' for tok in tokens):
                amr.alignments[node_id] = []
                print('[removing alignment]', 'you', f'({" ".join(tokens)})')
                changes += 1
        elif amr.nodes[node_id] == 'imperative':
            if not any(tok.startswith('let') or '!' in tok or tok=='imperative' for tok in tokens):
                amr.alignments[node_id] = []
                print('[removing alignment]', 'imperative', f'({" ".join(tokens)})')
                changes += 1
    amr.token2node_memo = {}
    # fix false negatives
    for node_id in amr.nodes:
        token_ids = amr.alignments[node_id]
        if token_ids:
            continue

        if amr.nodes[node_id] == '-':
            possible_align = [(i+1,tok) for i,tok in enumerate(amr.tokens) if tok.lower() in ['not','n\'t']]
            if len(possible_align)==1:
                id = possible_align[0][0]
                nodes = amr.alignmentsToken2Node(id)
                if len(nodes)==0:
                    amr.alignments[node_id] = [id]
                    print('[adding alignment]', '-', f'({possible_align[0][1]})')
                    changes += 1
    amr.token2node_memo = {}
    for node_id in amr.nodes:
        token_ids = amr.alignments[node_id]
        if not token_ids:
            continue
        nodes = amr.alignmentsToken2Node(token_ids[0])
        if len(nodes) <= 1:
            continue
        entity_sg = amr.findSubGraph(nodes)
        edges = entity_sg.edges
        tokens = [amr.tokens[t - 1] for t in token_ids if 0 <= t <= len(amr.tokens)]
        node_label = ','.join(amr.nodes[n] for n in nodes)
        if amr.nodes[node_id]=='name':
            for s,r,t in amr.edges:
                if (s,r,t) in edges:
                    continue
                if not r.startswith(':op'):
                    continue
                if node_id == s and t not in nodes:
                    amr.alignments[t] = amr.alignments[node_id]
                    print('[adding alignment]','name',amr.nodes[t], f'({" ".join(tokens)} {node_label})')
                    changes += 1
        root = entity_sg.root
        i = token_ids[-1]+1
        while i-1<len(amr.tokens):
            align = amr.alignmentsToken2Node(i)
            tok = amr.tokens[i-1]
            if not align and '"'+tok+'"' in [amr.nodes[n] for n in nodes] and tok not in tokens:
                for n in nodes:
                    amr.alignments[n].append(i)
                token_ids = amr.alignments[node_id]
                tokens = [amr.tokens[t - 1] for t in token_ids if 0 <= t <= len(amr.tokens)]
                print('[adding alignment]', 'token', tok, f'({" ".join(tokens)} {node_label})')
            else:
                break
            i+=1
        if not node_id == root:
            continue
        # for s, r, t in amr.edges:
        #     if (s, r, t) in edges:
        #         continue
        #     if node_id == s and t not in nodes and not amr.alignments[t]:
        #         amr.alignments[t].extend(amr.alignments[node_id])
        #         print('[adding alignment]', 'target', amr.nodes[t], f'({" ".join(tokens)} {node_label})')
        #         changes += 1
    amr.token2node_memo = {}
    return changes




def main():
    cr = JAMR_CorpusReader()
    cr.load_amrs(sys.argv[1], verbose=False)

    json = {'size':{}, 'unaligned':{}, 'unconnected':{}, 'unrooted':{}, 'repeats':{}, 'stats':{}}

    all_entities = []
    unaligned_nodes = []
    unrooted_entities = []
    changes = 0
    amrs_changed = 0
    for amr in cr.amrs:
        change = fix_alignments(amr)
        changes += change
        if change>0:
            amrs_changed+=1
        for node_id in amr.nodes:
            # get entity info
            if node_id not in amr.alignments:
                unaligned_nodes.append(amr.nodes[node_id])
                continue
            token_ids = amr.alignments[node_id]
            if not token_ids:
                unaligned_nodes.append(amr.nodes[node_id])
                continue
            nodes = amr.alignmentsToken2Node(token_ids[0])
            if len(nodes) <= 1:
                continue
            entity_sg = amr.findSubGraph(nodes)
            root = entity_sg.root
            if not node_id == root:
                continue
            edges = entity_sg.edges

            tokens = [amr.tokens[t-1] for t in token_ids if 0 <= t <= len(amr.tokens)]
            special_nodes = [n for n in nodes if (amr.nodes[n].isdigit() or amr.nodes[n].startswith('"'))]

            entity_type = sorted([amr.nodes[id] for id in nodes if id not in special_nodes])
            entity_type = ','.join(entity_type)

            nodes = {n: amr.nodes[n] for n in nodes}
            all_entities.append((amr, entity_type, tokens, root, nodes, edges, str(amr)))
            for s,r,t in amr.edges:
                if (s,r,t) in edges:
                    continue
                if len(edges)==0:
                    continue
                if s in nodes and s!=root:
                    if t not in amr.alignments or not amr.alignments[t]:
                        continue
                    label = f'{amr.nodes[root]} {amr.nodes[s]}'
                    unrooted_entities.append((entity_type, tokens, label, str(amr)))
                if t in nodes and t!=root:
                    if s not in amr.alignments or not amr.alignments[s]:
                        continue
                    label = f'{amr.nodes[root]} {amr.nodes[t]}'
                    unrooted_entities.append((entity_type, tokens, label, str(amr)))


    size_counters = dict()
    unconnected_counter = Counter()
    unaligned_counter = Counter()
    unrooted_counter = Counter()
    repeated_counter = Counter()
    attachment_counter = Counter()
    for node in unaligned_nodes:
        unaligned_counter[node] += 1
    for entity_type, tokens, label, string in unrooted_entities:
        unrooted_counter[entity_type]+=1
        attachment_counter[label] += 1
    json['stats']['unrooted-attachments'] = {}
    for node in sorted(attachment_counter, reverse=True, key=lambda x:attachment_counter[x]):
        json['stats']['unrooted-attachments'][node] = attachment_counter[node]
    for amr, entity_type, tokens, root, nodes, edges, string in all_entities:
        label = str(entity_type.count(',')+1)
        if label not in size_counters:
            size_counters[label] = Counter()
        size_counters[label][entity_type] += 1
        if entity_type.count(',')+1>1 and len(edges)==0:
            unconnected_counter[entity_type] += 1
        nodes = entity_type.split(',')
        if any(nodes.count(n)>1 for n in nodes):
            repeated_counter[entity_type]+=1

    print('Changes:',changes,'AMRs changed:',amrs_changed)
    for label in sorted(size_counters.keys(), key=lambda x:int(x)):
        print('size',label)
        print(f'({len(size_counters[label])} types, {sum(size_counters[label].values())} items)')
        json['stats']['size '+label] = {'types':len(size_counters[label]),
                                        'items':sum(size_counters[label].values())}
        print(size_counters[label])
        json['size'][label] = {}
        for type in sorted(size_counters[label],reverse=True,key=lambda x:size_counters[label][x]):
            d = {
                'count':size_counters[label][type],
                'tokens':[],
                'graphs':[],
            }
            json['size'][label][type] = d
    print('unconnected')
    print(f'({len(unconnected_counter)} types, {sum(unconnected_counter.values())} items)')
    json['stats']['unconnected'] = {'types': len(unconnected_counter),
                                      'items': sum(unconnected_counter.values())}
    print(unconnected_counter)
    json['unconnected'] = {}
    for type in sorted(unconnected_counter, reverse=True, key=lambda x: unconnected_counter[x]):
        d = {
            'count': unconnected_counter[type],
            'tokens': [],
            'graphs': [],
        }
        json['unconnected'][type] = d
    print('unaligned')
    print(f'({len(unaligned_counter)} types, {sum(unaligned_counter.values())} items)')
    json['stats']['unaligned'] = {'types': len(unaligned_counter),
                                    'items': sum(unaligned_counter.values())}
    print(unaligned_counter)
    json['unaligned'] = {}
    for type in sorted(unaligned_counter, reverse=True, key=lambda x: unaligned_counter[x]):
        d = {
            'count': unaligned_counter[type],
        }
        if type.isdigit():
            type = '<NUM>'+type
        json['unaligned'][type] = d
    print('unrooted')
    print(f'({len(unrooted_counter)} types, {sum(unrooted_counter.values())} items)')
    json['stats']['unrooted'] = {'types': len(unrooted_counter),
                                  'items': sum(unrooted_counter.values())}
    print(unrooted_counter)
    json['unrooted'] = {}
    for type in sorted(unrooted_counter, reverse=True, key=lambda x: unrooted_counter[x]):
        d = {
            'count': unrooted_counter[type],
            'tokens': [],
            'graphs': [],
            'attachments': []
        }
        json['unrooted'][type] = d
    print('repeats')
    print(f'({len(repeated_counter)} types, {sum(repeated_counter.values())} items)')
    json['stats']['repeats'] = {'types': len(repeated_counter),
                                  'items': sum(repeated_counter.values())}
    print(repeated_counter)
    json['repeats'] = {}
    for type in sorted(repeated_counter, reverse=True, key=lambda x: repeated_counter[x]):
        d = {
            'count': repeated_counter[type],
            'tokens': [],
            'graphs': [],
        }
        json['repeats'][type] = d
    print()


    for entity_type, tokens, label, string in unrooted_entities:
        tokens = ' '.join(tokens)
        if tokens not in json['unrooted'][entity_type]['tokens'] and len(json['unrooted'][entity_type]['tokens'])<100:
            json['unrooted'][entity_type]['tokens'].append(tokens)
            if len(json['unrooted'][entity_type]['graphs']) < 1:
                json['unrooted'][entity_type]['graphs'].append(string)
        if label not in json['unrooted'][entity_type]['attachments']:
            json['unrooted'][entity_type]['attachments'].append(label)
    for amr, entity_type, tokens, root, nodes, edges, string in all_entities:
        tokens = ' '.join(tokens)
        size = str(entity_type.count(',') + 1)
        if tokens not in json['size'][size][entity_type]['tokens'] and len(json['size'][size][entity_type]['tokens'])<100:
            json['size'][size][entity_type]['tokens'].append(tokens)
            if len(json['size'][size][entity_type]['graphs'])<1:
                json['size'][size][entity_type]['graphs'].append(string)
        if entity_type.count(',')+1>1 and len(edges)==0:
            if tokens not in json['unconnected'][entity_type]['tokens'] and len(json['unconnected'][entity_type]['tokens']) < 100:
                json['unconnected'][entity_type]['tokens'].append(tokens)
                if len(json['unconnected'][entity_type]['graphs']) < 1:
                    json['unconnected'][entity_type]['graphs'].append(string)
        nodes = entity_type.split(',')
        if any(nodes.count(n) > 1 for n in nodes):
            if tokens not in json['repeats'][entity_type]['tokens'] and len(json['repeats'][entity_type]['tokens']) < 100:
                json['repeats'][entity_type]['tokens'].append(tokens)
                if len(json['repeats'][entity_type]['graphs']) < 1:
                    json['repeats'][entity_type]['graphs'].append(string)

    with open('alignment_analysis.json', 'w+', encoding='utf8') as f:
        J.dump(json, f)



if __name__ == '__main__':
    main()
