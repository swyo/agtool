import os
import fire

from test_sampler import _get_sampler

from agtool.misc.profilling import timeit
from agtool.misc.log import get_logger, disable


logger = get_logger('helper_negative_sampling', level='INFO')
disable('DEBUG')


def run(dataset, batch_size, num_negatives, num_workers, epochs):
    os.makedirs('./cache', exist_ok=True)
    sampler = _get_sampler(dataset=dataset, batch_size=batch_size, num_negatives=num_negatives, num_workers=num_workers, supervised=False, cache=True)
    logger.info('Start helping negative sampling')
    for epoch in range(epochs):
        cache_fname = f'./cache/negatives-{epoch}'
        timeit(sampler.reset_negatives, cache_fname=cache_fname)
        logger.info(f'Cache negatives-{epoch} done')


if __name__ == '__main__':
    fire.Fire(run)
