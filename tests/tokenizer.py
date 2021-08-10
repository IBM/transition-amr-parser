import sys
import re
from transition_amr_parser.io import protected_tokenizer


# def protected_tokenizer(sentence_string):
#
#     # Do not split these strings
#     protected_re = re.compile(
#         r'[0-9][0-9,\.:/-]+[0-9]'          # quantities, time, dates
#         r'|[0-9]+[\.](?!\w)'               # enumerate
#         r'|(\W|^)[A-Za-z][\.](?!\w)'       # itemize
#         r'|\b([A-Z]\.)+'                   # acronym with periods (e.g. U.S.)
#         r'|!+|\?+|\.+|-+'                  # emphatic
#         r'|etc\.|i\.e\.|e\.g\.'            # latin abbreviations
#         r'|\b[Nn]o\.|\bUS\$|\b[Mm]r\.|\b[Mm]s\.'   # ...
#         r'|a\.m\.'                         # other abbreviations
#         r'|:\)|:\('                        # basic emoticons
#         # contractions
#         r'|[A-Za-z]+\'[A-Za-z]{3,}'        # quotes inside words
#         r'|n\'t(?!\w)'                     # negative contraction (needed?)
#         r'|\'m(?!\w)'                      # other contractions
#         r'|\'ve(?!\w)'                     # other contractions
#         r'|\'ll(?!\w)'                     # other contractions
#         r'|\'d(?!\w)'                      # other contractions
#         # r'|\'t(?!\w)'                      # other contractions
#         r'|\'re(?!\w)'                     # other contractions
#         r'|\'s(?!\w)'                      # saxon genitive
#         #
#         r'|<<|>>'                          # weird symbols
#     )
#
#     # otherwise split by these symbols
#     # TODO: Do we really need to split by - ?
#     sep_re = re.compile(r'[/~\*%\.,;:?!"\' \(\)\[\]\{\}-]')
#
#     # iterate over protected sequences, tokenize unprotected and append
#     # protected strings
#     tokens = []
#     positions = []
#     start = 0
#     for point in protected_re.finditer(sentence_string):
#
#         # extract preceeding and protected strings
#         end = point.start()
#         preceeding_str = sentence_string[start:end]
#         protected_str = sentence_string[end:point.end()]
#
#         if preceeding_str:
#             # tokenize preceeding string keep protected string as is
#             for token, (start2, end2) in zip(
#                 *simple_tokenizer(preceeding_str, sep_re)
#             ):
#                 tokens.append(token)
#                 positions.append((start + start2, start + end2))
#         tokens.append(protected_str)
#         positions.append((end, point.end()))
#
#         # move cursor
#         start = point.end()
#
#     # Termination
#     end = len(sentence_string)
#     if start < end:
#         ending_str = sentence_string[start:end]
#         if ending_str.strip():
#             for token, (start2, end2) in zip(
#                 *simple_tokenizer(ending_str, sep_re)
#             ):
#                 tokens.append(token)
#                 positions.append((start + start2, start + end2))
#
#     return tokens, positions
#
#
# def simple_tokenizer(sentence_string, separator_re):
#
#     tokens = []
#     positions = []
#     start = 0
#     for point in separator_re.finditer(sentence_string):
#
#         end = point.start()
#         token = sentence_string[start:end]
#         separator = sentence_string[end:point.end()]
#
#         # Add token if not empty
#         if token.strip():
#             tokens.append(token)
#             positions.append((start, end))
#
#         # Add separator
#         if separator.strip():
#             tokens.append(separator)
#             positions.append((end, point.end()))
#
#         # move cursor
#         start = point.end()
#
#     # Termination
#     end = len(sentence_string)
#     if start < end:
#         token = sentence_string[start:end]
#         if token.strip():
#             tokens.append(token)
#             positions.append((start, end))
#
#     return tokens, positions


if __name__ == '__main__':

    do_break = False

    # AMR file with ::snt and ::tok fields (JAMR)
    amr_file = sys.argv[1]

    tokens = []
    sents = []
    with open(amr_file) as fid:
        for line in fid:
            if line.strip().startswith('# ::snt'):
                sents.append(line.split('# ::snt')[-1].strip())
            elif line.strip().startswith('# ::tok'):
                tokens.append(line.split('# ::tok')[-1].strip())

    assert len(tokens) == len(sents)

    count = 0
    for index in range(len(tokens)):

        new_tokens = ' '.join(protected_tokenizer(sents[index])[0])

        if tokens[index] == new_tokens:
            count += 1
        elif do_break:
            print(tokens[index])
            print(new_tokens)
            import ipdb; ipdb.set_trace(context=30)
            protected_tokenizer(sents[index])

    perc = count * 100. / len(tokens)
    print(f'match {count}/{len(tokens)} {perc:.2f} %')
