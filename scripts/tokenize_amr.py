import argparse
from transition_amr_parser.io import protected_tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-amr", type=str, help="AMR file to be tokenized",
                        required=True)
    parser.add_argument("--simple", help="Use bare minimum tokenization",
                        action='store_true')
    return parser.parse_args()


def main(args):

    # read
    raw_amr = []
    with open(args.in_amr) as fid:
        for line in fid:
            raw_amr.append(line.rstrip())

    # append tok line, ignoring previously existing ones
    existing_tokenization = False
    out_raw_amr = []
    for line in raw_amr:
        if line.strip().startswith('# ::snt'):
            out_raw_amr.append(line)
            # get tokens and also append
            sentence = line.split('# ::snt')[-1].strip()
            tokens, _ = protected_tokenizer(sentence, args.simple)
            tokens_str = ' '.join(tokens)
            out_raw_amr.append(f'# ::tok {tokens_str}')
        elif line.strip().startswith('# ::tok'):
            # ignore existing tokens
            existing_tokenization = True
        else:
            out_raw_amr.append(line)

    if existing_tokenization:
        print('\nWARNING: Ignored existing tokenization in {args.in_amr}\n')

    # write
    with open(args.in_amr, 'w') as fid:
        for line in out_raw_amr:
            fid.write(f'{line}\n')


if __name__ == '__main__':
    main(parse_arguments())
