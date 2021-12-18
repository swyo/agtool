import time
import fire
from os import path as osp
from os import cpu_count
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit, Planetoid
from model import NeighborSampler, RawNeighborSampler

from agtool.misc.profilling import timeit

device = 'cpu'
first = True


def _get_sampler(batch_size, supervised, num_negatives, num_workers, dataset):
    global first
    assert dataset in ['Cora', 'Reddit', 'CiteSeer', 'Pubmed']
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    if dataset == 'Reddit':
        dataset = Reddit(path)
    else:
        dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    data.to(torch.device(device))
    train_size = sum(data.train_mask).item()
    test_size = sum(data.test_mask).item()
    if first:
        print({"train_size": train_size, "test_size": test_size})
        print("Select the test dataset for iteration")
        first = False
    sampler_kwargs = dict(
        edge_index=data.edge_index, batch_size=batch_size,
        sizes=[25, 10], shuffle=True, return_e_id=False,
        num_nodes=data.num_nodes, node_idx=data.test_mask,
        num_workers=cpu_count() * 3 // 4 if num_workers == 0 else num_workers,
    )
    if not supervised:
        sampler_kwargs.update({"num_negatives": num_negatives})
    if supervised:
        sampler = RawNeighborSampler(**sampler_kwargs)
    else:
        sampler = NeighborSampler(**sampler_kwargs)
    return sampler


def _iteration(sampler):
    for _ in tqdm(sampler, ncols=100):
        pass


def main(batch_size=32, supervised=True, num_negatives=5, num_workers=8, dataset='Cora', iteration=True):
    config = dict(batch_size=batch_size, supervised=supervised, num_workers=num_workers, dataset=dataset, num_negatives=0)
    if not supervised:
        config.update({"num_negatives": num_negatives})
    print(config)
    strip = "=" * 100
    print(f"{strip}\n\tTest01 - init sampler\n{strip}")
    # timeit(_get_sampler, **config)
    sampler = _get_sampler(**config)
    if iteration:
        print(f"{strip}\n\tTest02 - iteration of sampler\n{strip}")
        number = 3
        start = time.time()
        timeit(_iteration, sampler, number=number)
        elapsed = time.time() - start
        average = elapsed / float(number)
        print(f"Average elpased time for an iteration: {average:.6f} [sec]")
    if not supervised:
        number = 3
        start = time.time()
        timeit(sampler.reset_negatives, sampler.edge_index)
        elapsed = time.time() - start
        average = elapsed / float(number)
        print(f"Average negative sampling time for an iteration: {average:.6f} [sec]")


if __name__ == '__main__':
    fire.Fire(main)
