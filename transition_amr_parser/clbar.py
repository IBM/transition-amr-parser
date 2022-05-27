#    Copyright 2021 International Business Machines
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# This file is standalone and intended to be used as well separately of the
# repository, hence the attached license above.

import shutil
from collections import Counter
from datetime import datetime
# external module
import numpy as np


def red_background(string):
    return "\033[101m%s\033[0m" % string


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


def green_font(string):
    return "\033[92m%s\033[0m" % string


def print_log(module, string):
    """formats printing of log to stdout"""
    timestamp = str(datetime.now()).split('.')[0]
    print(f'{timestamp} [{module}] {string}')


def clbar(
    xy=None,  # list of (x, y) tuples or Counter
    x=None,
    y=None,
    ylim=(None, None),
    ncol=None,    # Max number of lines for display (defauly window size)
    # show only top and bottom values
    topx=None,
    botx=None,
    topy=None,
    boty=None,
    # normalize to sum to 1
    norm=False,
    xfilter=None,  # f(x) returns bool to not skip this example in display
    yform=None     # Function receiveing single y value returns string
):
    """Print data structure in command line"""
    # Sanity checks
    if x is None and y is None:
        if isinstance(xy, np.ndarray):
            labels = [f'{i}' for i in range(xy.shape[0])]
            xy = list(zip(labels, list(xy)))
        elif isinstance(xy, Counter):
            xy = [(str(x), y) for x, y in xy.items()]
        else:
            assert isinstance(xy, list), "Expected list of tuples"
            assert isinstance(xy[0], tuple), "Expected list of tuples"
    else:
        assert x is not None and y is not None
        assert isinstance(x, list)
        assert isinstance(y, list) or isinstance(y, np.ndarray)
        assert len(x) == len(list(y))
        xy = list(zip(x, y))

    # normalize
    if norm:
        z = sum([x[1] for x in xy])
        xy = [(k, v / z) for k, v in xy]
    # show only top x
    if topx is not None:
        xy = sorted(xy, key=lambda x: float(x[0]))[-topx:]
    if botx is not None:
        xy = sorted(xy, key=lambda x: float(x[0]))[:botx]
    if boty is not None:
        xy = sorted(xy, key=lambda x: x[1])[:boty]
    if topy is not None:
        xy = sorted(xy, key=lambda x: x[1])[-topy:]
    # print list of tuples
    # determine variables to fit data to command line
    x_data, y_data = zip(*xy)
    width = max([
        len(str(x)) if x is not None else len('None') for x in x_data
    ])
    number_width = max([len(f'{y}') for y in y_data])
    # max and min values
    if ylim[1] is not None:
        max_y_data = ylim[1]
    else:
        max_y_data = max(y_data)
    if ylim[0] is not None:
        min_y_data = ylim[0]
    else:
        min_y_data = min(y_data)
    # determine scaling factor from screen size
    data_range = max_y_data - min_y_data
    if ncol is None:
        ncol, _ = shutil.get_terminal_size((80, 20))
    max_size = ncol - width - number_width - 3
    scale = max_size / data_range

    # plot
    print()
    blank = ' '
    if yform:
        min_y_data_str = yform(min_y_data)
        print(f'{blank:<{width}}{min_y_data_str}')
    else:
        print(f'{blank:<{width}}{min_y_data}')
    for (x, y) in xy:

        # Filter example by x
        if xfilter and not xfilter(x):
            continue

        if y > max_y_data:
            # cropped bars
            num_col = int((ylim[1] - min_y_data) * scale)
            if num_col == 0:
                bar = ''
            else:
                half_width = (num_col // 2)
                if num_col % 2:
                    bar = '\u25A0' * (half_width - 1)
                    bar += '//'
                    bar += '\u25A0' * (half_width - 1)
                else:
                    bar = '\u25A0' * half_width
                    bar += '//'
                    bar += '\u25A0' * (half_width - 1)
        else:
            bar = '\u25A0' * int((y - min_y_data) * scale)
        if x is None:
            x = 'None'
        if yform:
            y = yform(y)
            print(f'{x:<{width}} {bar} {y}')
        else:
            print(f'{x:<{width}} {bar} {y}')
    print()
