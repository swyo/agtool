# GraphSAGE

paper: https://arxiv.org/pdf/1706.02216.pdf

```
python model.py --dataset=Cora --batch_size=64 --epochs=10 --device=cuda --num_workers=8 --lr=0.01 --interval=1
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
