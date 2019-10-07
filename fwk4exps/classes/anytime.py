import sys
import inspect


class Anytime(object):
    def __init__(self, fwk):
        """
        if mode = true, proceed as normal
        if mode = false, do not execute block
        """
        self.mode = fwk.anytime

    def __enter__(self):
        if not self.mode:
            # print 'Met block-skipping criterion ...'
            # Do some magic
            sys.settrace(lambda *args, **keys: None)
            frame = inspect.currentframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise

    def __exit__(self, type, value, traceback):
        # print 'Exiting context ...'
        return True
