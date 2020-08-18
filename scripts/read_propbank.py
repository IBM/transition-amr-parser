import sys
import glob
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def read_frame(xml_file):
    
    root = ET.parse(xml_file).getroot()
    propbank = {}
    for predicate in root.findall('predicate'):
        lemma = predicate.attrib['lemma']
        for roleset_data in predicate.findall('roleset'):
    
            # ID of the role e.g. run.01
            pred_id = roleset_data.attrib['id']
    
            # basic meta-data
            propbank[pred_id] = {
                'lemma': lemma,
                'description': roleset_data.attrib['name']
            } 
    
            # alias
            propbank[pred_id]['aliases'] = []
            for aliases in roleset_data.findall('aliases'):
                for alias in aliases:
                    propbank[pred_id]['aliases'].append(alias.text)
    
            # roles
            propbank[pred_id]['roles'] = {}
            for roles in roleset_data.findall('roles'):
                for role in roles:
                    if role.tag == 'note':
                        continue
                    number = role.attrib['n']
                    propbank[pred_id]['roles'][f'ARG{number}'] = role.attrib
    
            # examples
            propbank[pred_id]['examples'] = []
            for examples in roleset_data.findall('example'):
                sentence = examples.findall('text')
                assert len(sentence) == 1
                sentence = sentence[0].text
                tokens = [x.text for x in examples.findall('rel')]
                args = []
                for x in examples.findall('arg'):
                    args.append(x.attrib)
                    args[-1].update({'text': x.text})
                propbank[pred_id]['examples'].append({
                    'sentence': sentence,
                    'tokens': tokens,
                    'args': args 
                })


    return propbank


if __name__ == '__main__':
    
    # Argument handling
    in_propank_folder, out_json = sys.argv[1:] 

    # Read propbank into dict
    propbank = {}
    num_files = 0
    for xml_file in tqdm(glob.glob(f'{in_propank_folder}/*.xml')):
        propbank.update(read_frame(xml_file))
        num_files += 1
    if not num_files:
        print('No XML files found!')
        exit(1)

    num_preds = len(propbank)
    num_examples = sum([len(x['examples']) for x in propbank.values()])
    print(f'{num_files} files {num_preds} predicates {num_examples} examples read')

    # Write it into json
    with open(out_json, 'w') as fid:
        fid.write(json.dumps(propbank))
