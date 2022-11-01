# set this to true to start the debugger on any exception
DEBUG_MODE = False
if DEBUG_MODE:
    import sys
    import ipdb
    import traceback

    def debughook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        print()
        # post-mortem debugger
        ipdb.pm()
    sys.excepthook = debughook
