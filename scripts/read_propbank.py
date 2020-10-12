import sys
import glob
import json
from tqdm import tqdm
from transition_amr_parser.io import read_frame

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
