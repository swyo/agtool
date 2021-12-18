# GCN

```
python model.py

# result
Train: 100%|███████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 76.51it/s, Acc(val)=0.7940[<=0.7940], Acc(train)=0.943]
Load weights from gcn.pt
Test Accuracy: 0.804
```

Cora에 대한 sota 결과: https://paperswithcode.com/sota/node-classification-on-cora

## Comment

Transductive learning 으로 사용 되는데 모든 edge 정보를 넣지 않으면 성능저하가 발생함.
모든 정보를 메모리에 넣을 수 없는 경우가 생길 수 있음.
