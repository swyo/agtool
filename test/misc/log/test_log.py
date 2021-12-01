import random
import unittest

from agtool.misc.log import get_logger, disable

def _funcA(sub=None):
    num = random.randint(1, 10)
    loggerA = get_logger(f'_funcA-{num}', 'DEBUG')
    loggerA.debug(f'This is DEBUG[{num}]')
    if sub:
        sub()

def _funcB(sub=None):
    num = random.randint(1, 10)
    loggerB = get_logger(f'_funcB-{num}', 'INFO')
    loggerB.info(f'This is INFO[{num}]')
    if sub:
        sub()

def _funcC(sub=None):
    num = random.randint(1, 10)
    loggerC = get_logger(f'_funcC-{num}', 'WARN')
    loggerC.warning(f'This is WARN message[{num}]')
    if sub:
        sub()

class TestLog(unittest.TestCase):

    def test01_range(self):
        logger01 = get_logger('test01', 'INFO')
        logger01.info('This is info message')
        logger01.debug('This message should not be shown')
        logger01.warning('This is warn message')

    def test02_child(self):
        _funcA(_funcB)
        disable('INFO')
        _funcA(_funcB)
        _funcB(_funcC)
        disable('NOTSET')
        _funcA(_funcB)
        disable('INFO')
        _funcC(_funcB)
