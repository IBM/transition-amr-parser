import sys
import re
import json
from transition_amr_parser.io import read_amr
from collections import defaultdict
from tqdm import tqdm


def get_propbank_name(amr_pred):
    items = amr_pred.split('-')
    prop_pred = '-'.join(items[:-1]) + '.' + items[-1]
    if prop_pred.endswith('.91') or prop_pred in ['have-half-life.01']: 
        pass
    else:
        prop_pred = prop_pred.replace('-', '_')
    return prop_pred


if __name__ == '__main__':
    
    # Argument handling
    in_amr, in_propbank_json = sys.argv[1:] 

    corpus = read_amr(in_amr)
    with open(in_propbank_json) as fid:
        propbank = json.loads(fid.read())

    pred_regex = re.compile('.+-[0-9]+$')

    amr_alerts = defaultdict(list)
    sid = 0
    for amr in tqdm(corpus.amrs):
        predicate_ids = [
            k for k, v in amr.nodes.items() if pred_regex.match(v)
        ]
        for pred_id in predicate_ids:
            pred = get_propbank_name(amr.nodes[pred_id])
            if pred not in propbank:
                amr_alerts['predicate not in propbank'].append(
                    (sid, pred_id, pred)
                )
            else:    
                probank_roles = propbank[pred]['roles']
                # TODO: Identify obligatory args
#                 for k, v in probank_roles.items():
#                     if 'must' in v['descr']:
#                         import ipdb; ipdb.set_trace(context=30)
#                         print()
                roles = [
                    trip[1][1:].replace('-of', '') 
                    for trip in amr.edges 
                    if trip[0] == pred_id and trip[1].startswith(':ARG')
                ]
                forbidden_roles = set(roles) - set(probank_roles.keys())
                if forbidden_roles:
                    amr_alerts['role not in propbank'].append(
                        (sid, pred_id, pred, " ".join(list(forbidden_roles)))
                    )
            sid += 1

    import ipdb; ipdb.set_trace(context=30)
    print()
