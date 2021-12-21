# GraphSAGE

Paper: https://arxiv.org/pdf/1706.02216.pdf

Given node features and edge information, GraphSAGE outputs as follows.
- supervised: node classification
- unsupervised: node embedding

> Tip: `SparseTensor(edge_index[0], edge_index[1], sparse_size=(num_nodes, num_nodes)).sample_adj(batch, num_neighbors, replace=False)` can return (adj_t, n_id). <br>
> So, `model.full_forward(dataset[0].x[n_id], adj_t)[batch]` returns logit or embedding.

## Experiments

For cora
```
python model.py --dataset=Cora --batch_size=64 --epochs=10 --device=cuda --num_workers=8 --lr=0.01
GraphSAGE(
  (convs): ModuleList(
    (0): SAGEConv(1433, 256)
    (1): SAGEConv(256, 7)
  )
)
Train: 100%|█████████████████████████████████████████████████████████████| 10/10 [00:16<00:00,  1.69s/it, Acc(val)=0.7900[<=0.7900], Acc(train)=0.986]
Load weights from graphsage.pt
Test Accuracy: 0.802
```

For reddit
```
python model.py --dataset=Reddit --batch_size=64 --epochs=30 --device=cuda --num_workers=8 --lr=0.01
GraphSAGE(
  (convs): ModuleList(
    (0): SAGEConv(602, 256)
    (1): SAGEConv(256, 41)
  )
)
Train: 100%|████████████████████████████████████████████████████████████| 30/30 [50:18<00:00, 100.61s/it, Acc(val)=0.9293[<=0.9338], Acc(train)=0.878]
Load weights from graphsage.pt
Test Accuracy: 0.932337576073102
```

Negative_sampling Test

```
(agtool) root@docker-home:sage (models/gat !?*)$ python test.py                                                                                                                 
Option:  0
Epochs: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [47:25<00:00, 569.15s/it]
[INFO    ] 2021-12-10 11:59:48,696 [profilling.py] [timeit:46] [iteration] takes 2845.748234 [seconds]
Option:  1
Epochs: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [50:25<00:00, 605.10s/it]
[INFO    ] 2021-12-10 12:50:14,204 [profilling.py] [timeit:46] [iteration] takes 3025.506711 [seconds]
```


## Unsupervised

For Cora
```
python model.py --epochs=50 --num_workers=4 --supervised=False --lr=2e-3 --num_negatives=10 --weight_decay=5e-5
```

For Reddit
```
python model.py --epochs=20 --num_workers=4 --supervised=False --lr=2e-4 --num_negatives=10 --weight_decay=5e-5 --dataset=Reddit --batch_size=64 --device=cuda
GraphSAGE(
  (convs): ModuleList(
    (0): SAGEConv(602, 512)
    (1): SAGEConv(512, 256)
  )
)
NeighborSampler(sizes=[25, 10]): Elapsed time is 271.203309 [sec]
Train: 100%|█████████████████████████████████████████████| 20/20 [8:09:32<00:00, 1468.64s/it, Acc(val)=0.9358[<=0.9358], Acc(train)=0.919, Loss=0.776]
Load weights from graphsage.pt                                                                                                                        
Test Accuracy: 0.9342584779993897
```

## Sampler Test

* supervised sampler
50000 dataset.
```
python test_sampler.py --batch_size=128 --dataset='Reddit'
```
get_sampler: 28 sec <br>
iteration: 66 sec <br>

* unsupervised sampler
```
python test_sampler.py --batch_size=128 --dataset='Reddit' --supervised=False
{'batch_size': 128, 'supervised': False, 'num_workers': 8, 'dataset': 'Reddit', 'num_negatives': 5}
====================================================================================================
        Test01 - init sampler
====================================================================================================
{'train_size': 153431, 'test_size': 55703}
Select the test dataset for iteration
====================================================================================================
        Test02 - iteration of sampler
====================================================================================================
100%|█████████████████████████████████████████████████████████████| 436/436 [01:07<00:00,  6.48it/s]
100%|█████████████████████████████████████████████████████████████| 436/436 [01:06<00:00,  6.52it/s]
100%|█████████████████████████████████████████████████████████████| 436/436 [01:05<00:00,  6.64it/s]
[INFO    ] 2021-12-18 22:01:29,141 [profilling.py] [timeit:46] [_iteration] takes 20.867717 [seconds]
Average elpased time for an iteration: 6.623268 [sec]
[INFO    ] 2021-12-18 22:03:36,034 [profilling.py] [timeit:46] [reset_negatives] takes 126.892478 [seconds]
Average negative sampling time for an iteration: 42.297798 [sec]
```
get_sampler: 150 sec <br>
iteration: 6 sec <br>
negative_sampling: 42 sec

## Help Negative Sampling

Cache negative samples and use them.
```
python helper_negative_sampling.py Reddit 64 10 4 20
Activated logging levels: ('INFO', 'WARN', 'ERROR', 'CRITICAL')
{'train_size': 153431, 'test_size': 55703}
Select the train dataset for iteration
[INFO    ] 2021-12-19 11:57:07,395 [helper_negative_sampling.py] [run:17] Start helping negative sampling
[INFO    ] 2021-12-19 11:57:07,405 [profilling.py] [timeit:46] [reset_negatives] takes 255.009244 [seconds]
[INFO    ] 2021-12-19 11:57:07,405 [helper_negative_sampling.py] [run:21] Cache negatives-0 done
[INFO    ] 2021-12-19 12:01:22,971 [profilling.py] [timeit:46] [reset_negatives] takes 255.565027 [seconds]
[INFO    ] 2021-12-19 12:01:22,971 [helper_negative_sampling.py] [run:21] Cache negatives-1 done
[INFO    ] 2021-12-19 12:05:39,913 [profilling.py] [timeit:46] [reset_negatives] takes 256.941563 [seconds]
[INFO    ] 2021-12-19 12:05:39,914 [helper_negative_sampling.py] [run:21] Cache negatives-2 done
...
```

