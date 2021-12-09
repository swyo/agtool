# GCN

```
python model.py --epochs=100

# result
Train: 100%|███████████████████████████████████████████████████████████| 100/100 [00:38<00:00,  2.63it/s, Acc(val)=0.7620[<=0.7780], Acc(train)=0.971]
Load weights from gcn.pt
Test Accuracy: 0.773
Test Accuracy (full used): 0.8230000138282776
```

Cora에 대한 sota 결과: https://paperswithcode.com/sota/node-classification-on-cora

## Comment
Accuracy가 0.83으로 표기되어 있는데, 이 결과에 대해 의문점이 있음.

test 시에는 test 데이터에 대한 정보만 사용해야하는데 그렇게 하지 않았다. <br> 
모든 정보를 사용하면 0.82 정도 나옴.
