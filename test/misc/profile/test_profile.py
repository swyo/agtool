import unittest

from agtool.misc.profile import available_memory, timeit, memit


def _func(N):
    return [_ for _ in range(10 ** N)]


class TestProfile(unittest.TestCase):
    def test01_available_memory(self):
        available_memory()

    def test02_timeit(self):
        for i in range(3, 8):
            timeit(_func, i)

    def test03_memit(self):
        for i in range(5, 9):
            memit(_func, i, interval=1e-20)
