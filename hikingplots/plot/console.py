import sys
import time
from io import StringIO

from termcolor import colored


class StepIndicator(object):
    def __init__(self, message, delay_stdout=True, measure_time=True):
        self.message = message
        self.delay_stdout = delay_stdout
        self.measure_time = measure_time

    def __enter__(self):
        print(f"{self.message} ... ", end="", flush=True)
        if self.delay_stdout:
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        if self.delay_stdout:
            stdout = self._stringio.getvalue()
            del self._stringio
            sys.stdout = self._stdout
        else:
            stdout = ""

        if type is not None:
            print(colored("fail", "red"), end="")
        elif stdout:
            print(colored("ok (but with stdout)", "yellow"), end="")
        else:
            print(colored("ok", "green"), end="")

        if self.measure_time:
            print(f" ({self.end - self.start:.1f}s)")
        else:
            print("")

        if stdout:
            print(stdout, end="", flush=True)

    def print_info(self, info):
        print(f"{info} ... ", file=self._stdout, end="", flush=True)


class NoStepIndicator(StepIndicator):
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def print_info(self, info):
        pass
