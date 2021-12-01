from time import sleep
from os.path import exists
from functools import partial
import timeit as timeit_module
from concurrent.futures import ThreadPoolExecutor

import psutil


def timeit(original_fn, *args, number=1, verbose=True, name=None, fix_unit=False, **kwargs):
    """Profiling elapsed time.

    Args:
        original_fn: a func to run.
        number: set the number of run a given function.

    Kwargs:
        verbose: print elasped time.
        name: if verbose is True, print the name of func.
        fix_unit: fix unit.

    Example:
        >>> A = 10
        >>> timeit(func, A, number=1, verbose=True)
    """
    name = original_fn.__name__ if name is None else name
    elapsed = timeit_module.timeit(partial(original_fn, *args, **kwargs), number=number)
    unit = 'seconds'
    if not fix_unit:
        nominator = float(1)
        if elapsed < 1e-3:
            unit = 'ms'
            nominator = float(1000)
        elif elapsed < 1e-6:
            unit = 'us'
            nominator = float(1000 ** 2)
        elif elapsed < 1e-9:
            unit = 'ns'
            nominator = float(1000 ** 3)
        elapsed *= nominator
    if verbose:
        print(f"[{name}] takes {elapsed:.6f} [{unit}]")
    return elapsed


def available_memory(verbose=True):
    available = psutil.virtual_memory().available / (1024 ** 3)
    if verbose:
        print(f"available physical memory: {available} [GB]")
    return available


def memit(original_fn, *args, interval=1e-6, verbose=True, return_func=False, name=None, fix_unit=False, **kwargs):
    """Profiling used memory while runing a given func.

    Args:
        original_fn: a func to profile memory usages.

    kwargs:
        interval: sampling memory usage interval.
        verbose: print memory usages.
        return_func: return option to given func's return value.
        name: set a func name to print.
        fix_unit: fix unit.

    Example:
        >>> A = 10
        >>> memit(func, A, interval=1e-8, verbose=True)
    """
    name = original_fn.__name__ if name is None else name
    if not exists('/etc/os-release'):
        print("Get peak memory for only linux system.")
        return
    result = None
    start_usage = psutil.virtual_memory().used
    with ThreadPoolExecutor() as executor:
        monitor = _MemoryMonitor(interval)
        mem_thread = executor.submit(monitor.measure_usage)
        try:
            fn_thread = executor.submit(original_fn, *args, **kwargs)
            result = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            peak_memory = mem_thread.result()
            increment = peak_memory - start_usage
            unit = 'bytes'
            if not fix_unit:
                denominator = float(1)
                if 1024 < increment < 1024 ** 2:
                    unit = 'KiB'
                    denominator = float(1024)
                elif 1024 ** 2 < increment < 1024 ** 3:
                    unit = 'MiB'
                    denominator = float(1024 ** 2)
                elif 1024 ** 3 < increment:
                    unit = 'GiB'
                    denominator = float(1024 ** 3)
                peak_memory /= float(1024 ** 3)
                increment /= denominator
    if verbose:
        print(f"[{name}] peak memory: {peak_memory:.4f} [GiB], increment: {increment:.4f} [{unit}]")
    if return_func:
        return peak_memory, increment, result
    return peak_memory, increment


class _MemoryMonitor:
    """Memory moniter class.
    Note:
        Please see this article.
            https://medium.com/survata-engineering-blog/monitoring-memory-usage-of-a-running-python-program-49f027e3d1ba
    """
    def __init__(self, interval=0.1):
        self.keep_measuring = True
        self.interval = interval

    def measure_usage(self):
        max_usage = 0
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                psutil.virtual_memory().used
            )
            sleep(self.interval)
        return max_usage
