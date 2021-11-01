import argparse
import re


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-amr", type=str, help="AMR file to be tokenized",
                        required=True)
    parser.add_argument("--out-amr", type=str, help="Output AMR file.",
                        required=True)
    return parser.parse_args()


def protected_tokenizer(sentence_string):
    # imitates JAMR (97% sentece acc on AMR2.0)
    # split by these symbols
    # TODO: Do we really need to split by - ?
    sep_re = re.compile(r'[/~\*%\.,;:?!"\' \(\)\[\]\{\}-]')
    return jamr_like_tokenizer(sentence_string, sep_re)


def jamr_like_tokenizer(sentence_string, sep_re):

    # quote normalization
    sentence_string = sentence_string.replace('``', '"')
    sentence_string = sentence_string.replace("''", '"')
    sentence_string = sentence_string.replace("“", '"')

    # currency normalization
    #sentence_string = sentence_string.replace("£", 'GBP')

    # Do not split these strings
    protected_re = re.compile("|".join([
        # URLs (this conflicts with many other cases, we should normalize URLs
        # a priri both on text and AMR)
        #r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
        #
        r'[0-9][0-9,\.:/-]+[0-9]',         # quantities, time, dates
        r'^[0-9][\.](?!\w)',               # enumerate
        r'\b[A-Za-z][\.](?!\w)',           # itemize
        r'\b([A-Z]\.)+[A-Z]?',             # acronym with periods (e.g. U.S.)
        r'!+|\?+|-+|\.+',                  # emphatic
        r'etc\.|i\.e\.|e\.g\.|v\.s\.|p\.s\.|ex\.',     # latin abbreviations
        r'\b[Nn]o\.|\bUS\$|\b[Mm]r\.',     # ...
        r'\b[Mm]s\.|\bSt\.|\bsr\.|a\.m\.', # other abbreviations
        r':\)|:\(',                        # basic emoticons
        # contractions
        r'[A-Za-z]+\'[A-Za-z]{3,}',        # quotes inside words
        r'n\'t(?!\w)',                     # negative contraction (needed?)
        r'\'m(?!\w)',                      # other contractions
        r'\'ve(?!\w)',                     # other contractions
        r'\'ll(?!\w)',                     # other contractions
        r'\'d(?!\w)',                      # other contractions
        #r'\'t(?!\w)'                      # other contractions
        r'\'re(?!\w)',                     # other contractions
        r'\'s(?!\w)',                      # saxon genitive
        #
        r'<<|>>',                          # weird symbols
        #
        r'Al-[a-zA-z]+|al-[a-zA-z]+',      # Arabic article
        # months
        r'Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.'
    ]))

    # iterate over protected sequences, tokenize unprotected and append
    # protected strings
    tokens = []
    positions = []
    start = 0
    for point in protected_re.finditer(sentence_string):

        # extract preceeding and protected strings
        end = point.start()
        preceeding_str = sentence_string[start:end]
        protected_str = sentence_string[end:point.end()]

        if preceeding_str:
            # tokenize preceeding string keep protected string as is
            for token, (start2, end2) in zip(
                *simple_tokenizer(preceeding_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))
        tokens.append(protected_str)
        positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        ending_str = sentence_string[start:end]
        if ending_str.strip():
            for token, (start2, end2) in zip(
                *simple_tokenizer(ending_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))

    return tokens, positions


def simple_tokenizer(sentence_string, separator_re):

    tokens = []
    positions = []
    start = 0
    for point in separator_re.finditer(sentence_string):

        end = point.start()
        token = sentence_string[start:end]
        separator = sentence_string[end:point.end()]

        # Add token if not empty
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

        # Add separator
        if separator.strip():
            tokens.append(separator)
            positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        token = sentence_string[start:end]
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

    return tokens, positions


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

