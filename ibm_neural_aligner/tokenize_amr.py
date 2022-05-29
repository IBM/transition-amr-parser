import argparse
from transition_amr_parser.amr import protected_tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-amr", type=str, help="AMR file to be tokenized",
                        required=True)
    parser.add_argument("--out-amr", type=str, help="Output AMR file.",
                        required=True)
    return parser.parse_args()


def main(args):
    """
    Add `# ::tok` line with newly tokenized sentence.
    """

    # read and write
    with open(args.in_amr) as f_in, open(args.out_amr, 'w') as f_out:
        for line in f_in:

            if line.startswith('# ::tok'):
                raise Exception("File already tokenized!")

            elif line.startswith('# ::snt'):
                f_out.write(line)

                # tokenize
                sentence = line.split('# ::snt')[-1].strip()
                tokens, _ = protected_tokenizer(sentence)
                tokens_str = ' '.join(tokens)
                f_out.write(f'# ::tok {tokens_str}\n')

            else:
                f_out.write(line)


if __name__ == '__main__':
    main(parse_arguments())
