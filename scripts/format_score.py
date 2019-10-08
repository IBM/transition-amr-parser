# 
# Use as 
#
#    cat models/*/model_smatch.txt | python scripts/format_score.py
#

import re
import sys

result_regex = re.compile('.*F-score (0.[0-9]+)')

#  'black': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34,
#  'magenta': 35, 'cyan': 36, 'light gray': 37, 'dark gray': 90,
#  'light red': 91, 'light green': 92, 'light yellow': 93,
#  'light blue': 94, 'light magenta': 95, 'light cyan': 96, 'white': 97


def red(text):
    return "\033[%dm%s\033[0m" % (91, text)


def green(text):
    return "\033[%dm%s\033[0m" % (92, text)


if __name__ == '__main__':
    lines = sys.stdin
    prev_smatch = None
    best_smatch = None
    for line in lines:
        if result_regex.match(line):
            smatch = float(result_regex.match(line).groups()[0])
            if prev_smatch is not None:
                if best_smatch is None or best_smatch < smatch:
                    best = ""
                    best_smatch = smatch
                else:
                    best = "*"
                delta = smatch - prev_smatch
                if delta > 0:
                    print(line.strip() + green(" +%2.3f" % delta) + best)
                elif delta < 0:
                    print(line.strip() + red(" %2.3f" % delta) + best)
                else:    
                    print(line.strip() + best)
            else:
                print(line.strip())
            prev_smatch = smatch
        else:
            # New set of results
            prev_smatch = None
            best_smatch = None
            print(line.strip())
