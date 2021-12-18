# GraphSAGE

Paper: https://arxiv.org/pdf/1706.02216.pdf

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
python model.py --epochs=50 --interval=1 --num_workers=4 --supervised=False --lr=2e-3 --num_negatives=10 --weight_decay=5e-5


## Sampler Test

* supervised sampler
50000 dataset.
```
python test_sampler.py --batch_size=128 --dataset='Reddit'
```
get_sampler: 28 sec
iteration: 5 sec

* unsupervised sampler
get_sampler: 150 sec
iteration: 198 sec


