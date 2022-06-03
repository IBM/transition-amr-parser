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

    amrs = read_amr(in_amr)
    with open(in_propbank_json) as fid:
        propbank = json.loads(fid.read())

    pred_regex = re.compile('.+-[0-9]+$')

    amr_alerts = defaultdict(list)
    sid = 0
    num_preds = 0
    for amr in tqdm(amrs):
        predicate_ids = [
            k for k, v in amr.nodes.items() if pred_regex.match(v)
        ]
        num_preds += len(predicate_ids)
        for pred_id in predicate_ids:
            pred = get_propbank_name(amr.nodes[pred_id])
            if pred not in propbank:
                amr_alerts['predicate not in propbank'].append(
                    (sid, pred_id, pred)
                )
            else:
                probank_roles = propbank[pred]['roles']
                # TODO: Identify obligatory args
                required_roles = set()
                required_location = set()
                for k, v in probank_roles.items():
                    if '(must be specified)' in v['descr']:
                        required_roles |= set([k])
                    elif 'must' in v['descr']:
                        # FIXME: not used right now
                        required_location = set([k])

                # Get roles
                roles = [
                    trip[1][1:].replace('-of', '')
                    for trip in amr.edges
                    if trip[0] == pred_id and trip[1].startswith(':ARG')
                ]
                # Check no required missing
                missing_roles = required_roles - set(roles)
                if missing_roles:
                    amr_alerts['missing required role'].append(
                        (sid, pred_id, pred, " ".join(list(missing_roles)))
                    )
                # Check no forbiden used
                forbidden_roles = set(roles) - set(probank_roles.keys())
                if forbidden_roles:
                    amr_alerts['role not in propbank'].append(
                        (sid, pred_id, pred, " ".join(list(forbidden_roles)))
                    )
        sid += 1

    print(f'{sid+1} sentences {num_preds} predicates')
    for name, alerts in amr_alerts.items():
        if alerts:
            print(f'{len(alerts)} {name}')
