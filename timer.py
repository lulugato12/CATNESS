import time

class Timer(object):
    def __init__(self, msg = None):
        if msg:
            print(msg)

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, type, value, traceback):
        print('  Elapsed time: %.2f sec.' % (time.time() - self.tic))
