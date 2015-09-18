from time import time
from nlip.utils import hms as fhms

class Timer():
    """
    A simple timer class to measure time differences.

    Useful for producing reports at fixed intervals.

    Attributes
    ----------
    interval : int
        Interval at which `ready()` returns ``True``.
    elapsed : int
        Seconds since the last `tic()`.
    start : int
        epoch time at the last `tic()`.
    next_report : int
        epoch time of next report.

    Methods
    -------
    tic()
        Start counting.
    toc(hms=False)
        Return the time since the last call to `toc()`. If `hms` is set,
        return a string with ``HHhMMmSSs`` format.
    ready()
        Check if `self.interval` seconds have elapsed since the last call to
        `toc()`.

    """

    def __init__(self, interval=20):
        self.interval = interval

    def tic(self):
        self.start = time()
        self.next_report = self.interval
        self.elapsed = 0

    def toc(self, hms=False):
        self.elapsed = time() - self.start
        self.next_report = self.elapsed + self.interval
        if hms:
            return fhms(self.elapsed)
        return self.elapsed

    def ready(self):
        if time() - self.start >= self.next_report:
            return True
        return False

