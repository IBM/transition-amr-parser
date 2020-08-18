import re
import json as JSON
import sys
from collections import Counter
from transition_amr_parser.amr import JAMR_CorpusReader, AMR

entity_rules_json = {}

verbose = True


def create_fixed_rules(all_entities):
    global entity_rules_json

    fixed_counters = {}
    fixed_rules = {}

    for amr, entity_type, tokens, root, nodes, edges in all_entities:

        final_nodes = [n for n in nodes if not [e for e in edges if e[0] == n]]
        ntokens = [normalize(t) for t in tokens]

        if not all(t in [amr.nodes[n] for n in final_nodes] for t in ntokens):
            # normalize entity subgraph
            normalized_entity = normalize_entity(root, nodes, edges)
            name = ''.join(tokens).lower()
            if len(name) == 1 and name.isalpha():
                continue

            rule = entity_type + '\t' + ','.join(tokens).lower()
            node_string = str(normalized_entity.nodes)
            edge_string = str(normalized_entity.edges)
            if rule not in fixed_counters:
                fixed_counters[rule] = Counter()
                fixed_rules[rule] = {}
            solution_tag = node_string + '\t' + edge_string
            if solution_tag not in fixed_rules[rule]:
                fixed_rules[rule][solution_tag] = {}
            fixed_counters[rule][solution_tag] += 1
            fixed_rules[rule][solution_tag]['nodes'] = normalized_entity.nodes
            fixed_rules[rule][solution_tag]['edges'] = normalized_entity.edges
            # for s,r,t in normalized_entity.edges:
            #     if r==':quant' and not normalized_entity.nodes[t].isdigit():
            #         print(':quant',normalized_entity.nodes[t])
            #     if entity_type!='date-entity' and normalized_entity.nodes[t].isdigit():
            #         print(r,normalized_entity.nodes[t])
            fixed_rules[rule][solution_tag]['root'] = normalized_entity.root

    # clean rules
    for rule in sorted(list(fixed_counters.keys())):
        total = sum(fixed_counters[rule].values())
        name = rule.split('\t')[1]
        if len(name) < 3:
            if total < 3:
                del fixed_counters[rule]
                del fixed_rules[rule]
                # print('[deleting] '+rule)
                continue
        clean_rules(fixed_counters[rule], fixed_rules[rule])
        if verbose:
            print('[fixed rule]', rule)

    for country in countries:

        root = 0
        nodes = {0: 'country', 1: 'name', }
        edges = [(0, ':name', 1)]
        name = countries[country].split()
        idx = 2
        for n in name:
            nodes[idx] = f'"{n}"'
            edges.append((1, f':op{idx-1}', idx))
            idx += 1

        for rule in ['country,name' + '\t' + country, 'country,name' + '\t' + ','.join(name).lower()]:
            if rule in fixed_rules:
                continue
            else:
                fixed_rules[rule] = {}
                fixed_rules[rule]['count'] = 0
            fixed_rules[rule]['nodes'] = nodes
            fixed_rules[rule]['edges'] = edges
            fixed_rules[rule]['root'] = root
            if verbose:
                print('[fixed rule]', rule)

        root = 0
        nodes = {0: 'person', 1: 'country', 2: 'name'}
        edges = [(0, ':mod', 1), (1, ':name', 2)]
        name = countries[country].split()
        idx = 2
        for n in name:
            nodes[idx] = f'"{n}"'
            edges.append((2, f':op{idx - 1}', idx))
            idx += 1
        rule = 'person,country,name' + '\t' + ','.join(name).lower()
        if rule in fixed_rules:
            continue
        else:
            fixed_rules[rule] = {}
            fixed_rules[rule]['count'] = 0
        fixed_rules[rule]['nodes'] = nodes
        fixed_rules[rule]['edges'] = edges
        fixed_rules[rule]['root'] = root
        if verbose:
            print('[fixed rule]', rule)

    entity_rules_json['fixed'] = fixed_rules


def create_var_rules(all_entities):
    global entity_rules_json

    var_counters = {}
    var_rules = {}

    for amr, entity_type, tokens, root, nodes, edges in all_entities:
        # Update variable rules

        final_nodes = [n for n in nodes if not [e for e in edges if e[0] == n]]
        ntokens = [normalize(t) for t in tokens]

        if all(t in [amr.nodes[n] for n in final_nodes] for t in ntokens):
            if entity_type.endswith(',name') or entity_type == 'name':
                continue
            rule = entity_type + '\t' + str(len(ntokens))
            # normalize variables
            var_map = {}
            for idx, t in enumerate(ntokens):
                for n in nodes:
                    if t == amr.nodes[n]:
                        var_map[n] = f'X{idx}'
                    else:
                        var_map[n] = amr.nodes[n]
            # renormalize entity subgraph
            normalized_entity = normalize_entity(root, var_map, edges)
            node_string = str(normalized_entity.nodes)
            edge_string = str(normalized_entity.edges)
            if rule not in var_counters:
                var_counters[rule] = Counter()
                var_rules[rule] = {}
            solution_tag = node_string + '\t' + edge_string
            if solution_tag not in var_rules[rule]:
                var_rules[rule][solution_tag] = {}
            var_counters[rule][solution_tag] += 1
            var_rules[rule][solution_tag]['nodes'] = normalized_entity.nodes
            var_rules[rule][solution_tag]['edges'] = normalized_entity.edges
            var_rules[rule][solution_tag]['root'] = normalized_entity.root

    # clean rules
    for rule in sorted(list(var_counters.keys())):
        total = sum(var_counters[rule].values())
        # if total < 3:
        #     del var_counters[rule]
        #     del var_rules[rule]
        #     continue
        clean_rules(var_counters[rule], var_rules[rule])
        if verbose:
            print('[variable rule]', rule)
    rule = 'person\t1'
    root = 0
    nodes = {0: 'person', 1: 'X0'}
    edges = [(0, ':ARG0-of', 1)]
    if rule not in var_rules:
        var_rules[rule] = {}
        var_rules[rule]['count'] = 0
    var_rules[rule]['nodes'] = nodes
    var_rules[rule]['edges'] = edges
    var_rules[rule]['root'] = root
    if verbose:
        print('[variable rule]', rule)
    rule = 'thing\t1'
    root = 0
    nodes = {0: 'thing', 1: 'X0'}
    edges = [(0, ':ARG1-of', 1)]
    if rule not in var_rules:
        var_rules[rule] = {}
        var_rules[rule]['count'] = 0
    var_rules[rule]['nodes'] = nodes
    var_rules[rule]['edges'] = edges
    var_rules[rule]['root'] = root
    if verbose:
        print('[variable rule]', rule)

    entity_rules_json['var'] = var_rules


def create_date_entity_rules(all_entities):
    global entity_rules_json

    date_entity = {}

    for amr, entity_type, tokens, root, nodes, edges in all_entities:

        if entity_type == 'date-entity':
            final_nodes = [n for n in nodes if not [e for e in edges if e[0] == n]]
            final_edges = [e for e in edges if e[2] in final_nodes]
            for s, r, t in final_edges:
                if r not in date_entity:
                    date_entity[r] = []
                if amr.nodes[t] not in date_entity[r]:
                    date_entity[r].append(amr.nodes[t])
        if ':era' not in date_entity:
            date_entity[':era'] = []
        for t in ['BC', 'AD', 'B.C.', 'A.D.', 'CE', 'BCE', 'C.E.', 'B.C.E.', 'Heisei', 'Reiwa']:
            if t not in date_entity[':era']:
                date_entity[':era'].append(t)
        if ':dayperiod' not in date_entity:
            date_entity[':dayperiod'] = []
        for t in ['and', 'yesterday', 'today']:
            if t in date_entity[':dayperiod']:
                date_entity[':dayperiod'].remove(t)
    for rule in date_entity:
        if verbose:
            print('[date-entity rule]', rule)
    entity_rules_json['date-entity'] = date_entity


def create_name_rules(all_entities):
    global entity_rules_json

    name_rules = {}

    for amr, entity_type, tokens, root, nodes, edges in all_entities:

        final_nodes = [n for n in nodes if not [e for e in edges if e[0] == n]]
        rule_nodes = [id for id in nodes if id not in final_nodes]
        subgraph = amr.findSubGraph(rule_nodes)

        # if len(rule_nodes) < 3:
        #     continue

        if entity_type.endswith(',name') or entity_type == 'name':

            # renormalize entity subgraph
            normalized_entity = normalize_entity(subgraph.root, subgraph.nodes, subgraph.edges)

            rule = entity_type
            if rule not in name_rules:
                name_rules[rule] = {}
            name_rules[rule]['nodes'] = normalized_entity.nodes
            name_rules[rule]['edges'] = normalized_entity.edges
            name_rules[rule]['root'] = normalized_entity.root
    for rule in name_rules:
        if verbose:
            print('[names rule]', rule)
    entity_rules_json['names'] = name_rules


def clean_rules(counter, rule):
    max_val = max(counter.keys(), key=lambda x: counter[x])
    for solution in list(counter.keys()):
        if solution != max_val:
            del counter[solution]
            del rule[solution]
    for solution in counter:
        rule['nodes'] = rule[solution]['nodes']
        rule['edges'] = rule[solution]['edges']
        rule['root'] = rule[solution]['root']
        rule['count'] = counter[solution]
        del rule[solution]


def normalize_entity(root, nodes, edges):
    normalize_ids = {id: i for i, id in enumerate(sorted(nodes, key=lambda x: nodes[x]))}
    normalized_entity = AMR()
    for n in nodes:
        normalized_entity.nodes[normalize_ids[n]] = nodes[n]
    for s, r, t in edges:
        normalized_entity.edges.append((normalize_ids[s], r, normalize_ids[t]))
    normalized_entity.edges = sorted(normalized_entity.edges)
    normalized_entity.root = normalize_ids[root]
    return normalized_entity


def main():
    cr = JAMR_CorpusReader()
    cr.load_amrs(sys.argv[1], verbose=False)

    all_entities = []
    for amr in cr.amrs:
        for node_id in amr.alignments:

            # get entity info
            token_ids = amr.alignments[node_id]
            if not token_ids:
                continue
            nodes = amr.alignmentsToken2Node(token_ids[0])
            if len(nodes) <= 1:
                continue
            entity_sg = amr.findSubGraph(nodes)
            root = entity_sg.root
            if not node_id == root:
                continue
            edges = entity_sg.edges
            if not edges:
                continue
            if len(edges) == 1 and edges[0][1] in [':polarity', ':mode']:
                continue

            tokens = [amr.tokens[t-1] for t in token_ids if 0 <= t <= len(amr.tokens)]
            final_nodes = [n for n in nodes if not [e for e in edges if e[0] == n]]

            entity_type = [amr.nodes[id] for id in nodes if id not in final_nodes]
            entity_type = ','.join(entity_type)

            nodes = {n: amr.nodes[n] for n in nodes}
            all_entities.append((amr, entity_type, tokens, root, nodes, edges))

    create_fixed_rules(all_entities)
    create_var_rules(all_entities)
    create_name_rules(all_entities)
    create_date_entity_rules(all_entities)
    create_normalization_rules()

    if verbose:
        print('[entity rules] Writing rules')
    frules_out = sys.argv[2]
    with open(frules_out, 'w+', encoding='utf8') as f:
        JSON.dump(entity_rules_json, f, sort_keys=True)
    if verbose:
        print('[entity rules] Fixed:', len(entity_rules_json['fixed']))
        print('[entity rules] Variable:', len(entity_rules_json['var']))
        print('[entity rules] Date-entity:', len(entity_rules_json['date-entity']))
        print('[entity rules] Named entity:', len(entity_rules_json['names']))
        print('[entity rules] Normalize:', sum(len(x) for x in entity_rules_json['normalize'].values()))
        print('[entity rules] Done')


def normalize(string):
    lstring = string.lower()

    # number or ordinal
    if NUM_RE.match(lstring):
        return lstring.replace(',', '').replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')

    # months
    if lstring in months:
        return str(months[lstring])
    if len(lstring) == 4 and lstring.endswith('.') and lstring[:3] in months:
        return str(months[lstring[:3]])

    # cardinal numbers
    if lstring in cardinals:
        return str(cardinals[lstring])

    # ordinal numbers
    if lstring in ordinals:
        return str(ordinals[lstring])

    # unit abbreviations
    if lstring in units:
        return str(units[lstring])
    if lstring.endswith('s') and lstring[:-1] in units:
        return str(units[lstring[:-1]])
    if lstring in units.values():
        return lstring
    if string.endswith('s') and lstring[:-1] in units.values():
        return lstring[:-1]

    return '"' + string + '"'


def create_normalization_rules():
    entity_rules_json['normalize'] = {}
    entity_rules_json['normalize']['months'] = months
    entity_rules_json['normalize']['units'] = units
    entity_rules_json['normalize']['cardinals'] = cardinals
    entity_rules_json['normalize']['ordinals'] = ordinals
    entity_rules_json['normalize']['decades'] = decades
    entity_rules_json['normalize']['countries'] = countries
    print('[normalize rules] months')
    print('[normalize rules] units')
    print('[normalize rules] cardinals')
    print('[normalize rules] ordinals')


NUM_RE = re.compile(r'^([0-9]|,)+(st|nd|rd|th)?$')
months = {'january': 1,
          'february': 2,
          'march': 3,
          'april': 4,
          'may': 5,
          'june': 6,
          'july': 7,
          'august': 8,
          'september': 9,
          'october': 10,
          'november': 11,
          'december': 12,
          'jan': 1,
          'feb': 2,
          'mar': 3,
          'apr': 4,
          'jun': 6,
          'jul': 7,
          'aug': 8,
          'sep': 9,
          'oct': 10,
          'nov': 11,
          'dec': 12
          }
cardinals = {'one': 1,
             'two': 2,
             'three': 3,
             'four': 4,
             'five': 5,
             'six': 6,
             'seven': 7,
             'eight': 8,
             'nine': 9,
             'ten': 10,
             'eleven': 11,
             'twelve': 12,
             'thirteen': 13,
             'fourteen': 14,
             'fifteen': 15,
             'sixteen': 16,
             'seventeen': 17,
             'eighteen': 18,
             'nineteen': 19,
             'twenty': 20,
             'thirty': 30,
             'forty': 40,
             'fifty': 50,
             'sixty': 60,
             'seventy': 70,
             'eighty': 80,
             'ninety': 90,
             'hundred': 100,
             'thousand': 1000,
             'million': 1000000,
             'billion': 1000000000,
             'trillion': 1000000000000,
             }
ordinals = {'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10,
            'eleventh': 11,
            'twelfth': 12,
            'thirteenth': 13,
            'fourteenth': 14,
            'fifteenth': 15,
            'sixteenth': 16,
            'seventeenth': 17,
            'eighteenth': 18,
            'nineteenth': 19,
            'twentieth': 20,
            'thirtieth': 30,
            'fortieth': 40,
            'fiftieth': 50,
            'sixtieth': 60,
            'seventieth': 70,
            'eightieth': 80,
            'ninetieth': 90,
            'hundredth': 100,
            'thousandth': 1000,
            'millionth': 1000000,
            'billionth': 1000000000}
decades = {'twenties': 1920,
           'thirties': 1930,
           'forties': 1940,
           'fifties': 1950,
           'sixties': 1960,
           'seventies': 1970,
           'eighties': 1980,
           'nineties': 1990}
units = {'km': 'kilometer',
         'm': 'meter',
         'dm': 'decimeter',
         'cm': 'centimeter',
         'mm': 'millimeter',
         'mi': 'mile',
         'in': 'inch',
         'ft': 'foot',
         'yd': 'yard',
         'acre': 'acre',
         'hectare': 'hectare',
         't': 'tonne',
         'kg': 'kilogram',
         'hg': 'hectogram',
         'g': 'gram',
         'dg': 'decigram',
         'cg': 'centigram',
         'mg': 'milligram',
         # 'kmph': 'kilometer per hour',
         # 'mps': 'meter per second',
         # 'mph': 'mile per hour',
         # 'km/h': 'kilometer per hour',
         # 'm/s': 'meter per second',
         # 'mi/h': 'mile per hour',
         'l': 'liter',
         'lb': 'pound',
         'ml': 'milliliter',
         'oz': 'ounce',
         'pt': 'pint',
         'tsp': 'teaspoon',
         'tbsp': 'tablespoon',
         'cup': 'cup',
         'gal': 'gallon',
         'qt': 'quart',
         'h': 'hour',
         'hr': 'hour',
         'min': 'minute',
         's': 'second',
         'ms': 'millisecond',
         'day': 'day',
         'month': 'month',
         'week': 'week',
         'yr': 'year',
         'hz': 'Hertz',
         'khz': 'Kilohertz',
         'mhz': 'Megahertz',
         'ghz': 'Gigahertz',
         '°c': 'Celsius',
         '°f': 'Fahrenheit',
         'k': 'Kelvin',
         'b': 'byte',
         'kb': 'kilobyte',
         'mb': 'megabyte',
         'gb': 'gigabyte',
         'tb': 'terabyte',
         'pb': 'petabyte',
         'eb': 'exabyte',
         'zb': 'zettabyte',
         'yb': 'yottabyte',
         '$': 'dollar',
         '¢': 'cent',
         '£': 'pound',
         '¥': 'yen',
         '฿': 'baht',
         '₩': 'won',
         '€': 'euro',
         '₱': 'peso',
         '₹': 'rupee',
         '₿': 'bitcoin',
         'yuan': 'yuan',
         }
countries = {
    'abkhaz': 'Abkhazia',
    'abkhazian': 'Abkhazia',
    'afghan': 'Afghanistan',
    'åland,island': 'Åland,Islands',
    'aland,island': 'Åland,Islands',
    'albanian': 'Albania',
    'algerian': 'Algeria',
    'american': 'United States',
    'american,samoan': 'American Samoa',
    'andorran': 'Andorra',
    'angolan': 'Angola',
    'anguillan': 'Anguilla',
    'antarctic': 'Antarctica',
    'antiguan': 'Antigua and Barbuda',
    'barbudan': 'Antigua and Barbuda',
    'argentinian': 'Argentina',
    'armenian': 'Armenia',
    'aruban': 'Aruba',
    'australian': 'Australia',
    'austrian': 'Austria',
    'azerbaijani': 'Azerbaijan',
    'azeri': 'Azerbaijan',
    'bahamian': 'The Bahamas',
    'bahraini': 'Bahrain',
    'bangladeshi': 'Bangladesh',
    'barbadian': 'Barbados',
    'belarusian': 'Belarus',
    'belgian': 'Belgium',
    'belizean': 'Belize',
    'beninese': 'Benin',
    'beninois': 'Benin',
    'bermudian': 'Bermuda',
    'bermudan': 'Bermuda',
    'bhutanese': 'Bhutan',
    'bolivian': 'Bolivia',
    'bonaire': 'Bonaire',
    'herzegovinian': 'Bosnia and Herzegovina',
    'bosnian': 'Bosnia and Herzegovina',
    'motswana': 'Botswana',
    'botswanan': 'Botswana',
    'bouvet,island': 'Bouvet Island',
    'brazilian': 'Brazil',
    'british': 'United Kingdom',
    'biot': 'British Indian Ocean Territory',
    'bruneian': 'Brunei',
    'bulgarian': 'Bulgaria',
    'burkinabé': 'Burkina Faso',
    # 'burmese':'Burma',
    'burundian': 'Burundi',
    'cabo,verdean': 'Cabo Verde',
    'cambodian': 'Cambodia',
    'cameroonian': 'Cameroon',
    'canadian': 'Canada',
    'caymanian': 'Cayman Islands',
    'central,african': 'Central African Republic',
    'chadian': 'Chad',
    'chilean': 'Chile',
    'chinese': 'China',
    'christmas,island': 'Christmas Island',
    'cocos,island': 'Cocos Islands',
    'colombian': 'Colombia',
    'comorian': 'Comoros',
    'comoran': 'Comoros',
    'Congolese': 'Democratic Republic of the Congo',
    'cook,island': 'Cook Islands',
    'costa,rican': 'Costa Rica',
    'ivorian': 'Côte d\'Ivoire',
    'croatian': 'Croatia',
    'cuban': 'Cuba',
    'curaçaoan': 'Curaçao',
    'cypriot': 'Cyprus',
    'czech': 'Czech Republic',
    'danish': 'Denmark',
    'djiboutian': 'Djibouti',
    # 'dominican':	'Dominica',
    'dominican': 'Dominican Republic',
    'timorese': 'East Timor',
    'ecuadorian': 'Ecuador',
    'egyptian': 'Egypt',
    'salvadoran': 'El Salvador',
    'equatorial,guinean': 'Equatorial Guinea',
    'equatoguinean': 'Equatorial Guinea',
    'eritrean': 'Eritrea',
    'estonian': 'Estonia',
    # 'swazi':'Eswatini',
    # 'swati':'Eswatini',
    'ethiopian': 'Ethiopia',
    'european': 'European Union',
    'falkland,island': 'Falkland Islands',
    'faroese': 'Faroe Islands',
    'fijian': 'Fiji',
    'finnish': 'Finland',
    'french': 'France',
    'french,guianese': 'French Guiana',
    'french,polynesian': 'French Polynesia',
    'french,southern,territories': 'French Southern Territories',
    'gabonese': 'Gabon',
    'gambian': 'The Gambia',
    'georgian': 'Georgia',
    'german': 'Germany',
    'ghanaian': 'Ghana',
    'gibraltar': 'Gibraltar',
    'greek': 'Greece',
    'hellenic': 'Greece',
    'greenlandic': 'Greenland',
    'grenadian': 'Grenada',
    'guadeloupe': 'Guadeloupe',
    'guamanian': 'Guam',
    'guatemalan': 'Guatemala',
    # 'channel,island':	'Guernsey',
    'guinean': 'Guinea',
    'bissau-guinean': 'Guinea-Bissau',
    'guyanese': 'Guyana',
    'haitian': 'Haiti',
    'heard,island': 'Heard Island',
    'mcdonald,islands': 'Heard Island',
    'honduran': 'Honduras',
    'hong,kong': 'Hong Kong',
    'cantonese': 'Hong Kong',
    'hungarian': 'Hungary',
    'magyar': 'Hungary',
    'icelandic': 'Iceland',
    'indian': 'India',
    'indonesian': 'Indonesia',
    'iranian': 'Iran',
    'persian': 'Iran',
    'iraqi': 'Iraq',
    'irish': 'Ireland',
    'manx': 'Isle of Man',
    'israeli': 'Israel',
    'israelite': 'Israel',
    'italian': 'Italy',
    # 'ivorian':	'Ivory Coast',
    'jamaican': 'Jamaica',
    'jan,mayen': 'Jan Mayen',
    'japanese': 'Japan',
    'channel,island': 'Jersey',
    'jordanian': 'Jordan',
    'kazakhstani': 'Kazakhstan',
    'kazakh': 'Kazakhstan',
    'kenyan': 'Kenya',
    'i-kiribati': 'Kiribati',
    'north,korean': 'North Korea',
    'south,korean': 'South Korea',
    'kosovar': 'Kosovo',
    'kosovan': 'Kosovo',
    'kuwaiti': 'Kuwait',
    'kyrgyzstani': 'Kyrgyzstan',
    'kirgiz': 'Kyrgyzstan',
    'kirghiz': 'Kyrgyzstan',
    'kyrgyz': 'Kyrgyzstan',
    'lao': 'Laos',
    'laotian': 'Laos',
    'latvian': 'Latvia',
    'lettish': 'Latvia',
    'lebanese': 'Lebanon',
    'basotho': 'Lesotho',
    'liberian': 'Liberia',
    'libyan': 'Libya',
    'liechtensteiner': 'Liechtenstein',
    'lithuanian': 'Lithuania',
    'luxembourg': 'Luxembourg',
    'luxembourgish': 'Luxembourg',
    'macanese': 'Macau',
    'macedonian': 'Republic of North Macedonia',
    'malagasy': 'Madagascar',
    'malawian': 'Malawi',
    'malaysian': 'Malaysia',
    'maldivian': 'Maldives',
    'malian': 'Mali',
    'malinese': 'Mali',
    'maltese': 'Malta',
    'marshallese': 'Marshall Islands',
    'martiniquais': 'Martinique',
    'martinican': 'Martinique',
    'mauritanian': 'Mauritania',
    'mauritian': 'Mauritius',
    'mahoran': 'Mayotte',
    'mexican': 'Mexico',
    'micronesian': 'Federated States of Micronesia',
    'moldovan': 'Moldova',
    'monégasque': 'Monaco',
    'monacan': 'Monaco',
    'mongolian': 'Mongolia',
    'montenegrin': 'Montenegro',
    'montserratian': 'Montserrat',
    'moroccan': 'Morocco',
    'mozambican': 'Mozambique',
    'burmese': 'Myanmar',
    'namibian': 'Namibia',
    'nauruan': 'Nauru',
    'nepalese': 'Nepal',
    'nepali': 'Nepal',
    'netherlandic': 'Netherlands',
    'dutch': 'Netherlands',
    'new,caledonian': 'New Caledonia',
    'new,zealand': 'New Zealand',
    'nicaraguan': 'Nicaragua',
    'nigerien': 'Niger',
    'nigerian': 'Nigeria',
    'niuean': 'Niue',
    'norfolk,island': 'Norfolk Island',
    'northern,irish': 'Northern Ireland',
    'northern,marianan': 'Northern Mariana Islands',
    'norwegian': 'Norway',
    'omani': 'Oman',
    'pakistani': 'Pakistan',
    'palauan': 'Palau',
    'palestinian': 'Palestine',
    'panamanian': 'Panama',
    'papua,new,guinean': 'Papua New Guinea',
    'papuan': 'Papua New Guinea',
    'paraguayan': 'Paraguay',
    'peruvian': 'Peru',
    'filipino': 'Philippines',
    'philippine': 'Philippines',
    'pitcairn,island': 'Pitcairn Islands',
    'polish': 'Poland',
    'portuguese': 'Portugal',
    'puerto,rican': 'Puerto Rico',
    'qatari': 'Qatar',
    'réunionese': 'Réunion',
    'réunionnais': 'Réunion',
    'romanian': 'Romania',
    'russian': 'Russia',
    'rwandan': 'Rwanda',
    'saba': 'Saba',
    'barthélemois': 'Saint Barthélemy',
    'saint,helenian': 'Saint Helena',
    'kittitian': 'Saint Kitts and Nevis',
    'nevisian': 'Saint Kitts and Nevis',
    'saint,lucian': 'Saint Lucia',
    'saint-martinoise': 'Saint Martin',
    'saint-pierrais': 'Saint Pierre and Miquelon',
    'miquelonnais': 'Saint Pierre and Miquelon',
    'saint,vincentian': 'Saint Vincent and the Grenadines',
    'vincentian': 'Saint Vincent and the Grenadines',
    'samoan': 'Samoa',
    'sammarinese': 'San Marino',
    'são toméan': 'São Tomé and Príncipe',
    'saudi': 'Saudi Arabia',
    'saudi,arabian': 'Saudi Arabia',
    'scottish': 'Scotland',
    'senegalese': 'Senegal',
    'serbian': 'Serbia',
    'seychellois': 'Seychelles',
    'sierra,leonean': 'Sierra Leone',
    'singaporean': 'Singapore',
    'singapore': 'Singapore',
    'sint,eustatius': 'Sint Eustatius',
    'statian': 'Sint Eustatius',
    'sint,maarten': 'Sint Maarten',
    'slovak': 'Slovakia',
    'slovenian': 'Slovenia',
    'slovene': 'Slovenia',
    'solomon,island': 'Solomon Islands',
    'somali': 'Somalia',
    'somalilander': 'Somaliland',
    'south,african': 'South Africa',
    'south,georgia': 'South Georgia and the South Sandwich Islands',
    'south,sandwich,islands': 'South Georgia and the South Sandwich Islands',
    'south,ossetian': 'South Ossetia',
    'south,sudanese': 'South Sudan',
    'spanish': 'Spain',
    'sri,lankan': 'Sri Lanka',
    'sudanese': 'Sudan',
    'surinamese': 'Suriname',
    'svalbard': 'Svalbard',
    'swazi': 'Swaziland',
    'swati': 'Swaziland',
    'swedish': 'Sweden',
    'swiss': 'Switzerland',
    'syrian': 'Syria',
    'taiwanese': 'Taiwan',
    'formosan': 'Taiwan',
    'tajikistani': 'Tajikistan',
    'tanzanian': 'Tanzania',
    'thai': 'Thailand',
    # 'timorese':	'Timor-Leste',
    'togolese': 'Togo',
    'tokelauan': 'Tokelau',
    'tongan': 'Tonga',
    'trinidadian': 'Trinidad and Tobago',
    'tobagonian': 'Trinidad and Tobago',
    'tunisian': 'Tunisia',
    'turkish': 'Turkey',
    'turkmen': 'Turkmenistan',
    'turks,and,caicos,island': 'Turks and Caicos Islands',
    'tuvaluan': 'Tuvalu',
    'ugandan': 'Uganda',
    'ukrainian': 'Ukraine',
    'emirati': 'United Arab Emirates',
    'emirian': 'United Arab Emirates',
    'emiri': 'United Arab Emirates',
    'uruguayan': 'Uruguay',
    'uzbekistani': 'Uzbekistan',
    'uzbek': 'Uzbekistan',
    'ni-vanuatu': 'Vanuatu',
    'vanuatuan': 'Vanuatu',
    'vatican': 'Vatican City State',
    'venezuelan': 'Venezuela',
    'vietnamese': 'Vietnam',
    'british,virgin,island': 'British Virgin Islands',
    'u.s.,virgin,island': 'U.S. Virgin Islands',
    'wallis,and,futuna': 'Wallis and Futuna',
    'wallisian': 'Wallis and Futuna',
    'futunan': 'Wallis and Futuna',
    'sahrawi': 'Western Sahara',
    'sahrawian': 'Western Sahara',
    'sahraouian': 'Western Sahara',
    'yemeni': 'Yemen',
    'zambian': 'Zambia',
    'zimbabwean': 'Zimbabwe',
}


if __name__ == '__main__':
    main()
