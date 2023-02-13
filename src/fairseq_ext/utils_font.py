# FONT_COLORORS
FONT_COLOR = {
    'black': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34,
    'magenta': 35, 'cyan': 36, 'light gray': 37, 'dark gray': 90,
    'light red': 91, 'light green': 92, 'light yellow': 93,
    'light blue': 94, 'light magenta': 95, 'light cyan': 96, 'white': 97
}

# BG FONT_COLORORS
BACKGROUND_COLOR = {
    'black': 40, 'red': 41, 'green': 42, 'yellow': 43, 'blue': 44,
    'magenta': 45, 'cyan': 46, 'light gray': 47, 'dark gray': 100,
    'light red': 101, 'light green': 102, 'light yellow': 103,
    'light blue': 104, 'light magenta': 105, 'light cyan': 106,
    'white': 107
}


def white_background(string):
    return "\033[107m%s\033[0m" % string


def red_background(string):
    return "\033[101m%s\033[0m" % string


def black_font(string):
    return "\033[30m%s\033[0m" % string


def yellow_font(string):
    return "\033[93m%s\033[0m" % string


def stack_style(string):
    return black_font(white_background(string))


def ordered_exit(signum, frame):
    print("\nStopped by user\n")
    exit(0)
