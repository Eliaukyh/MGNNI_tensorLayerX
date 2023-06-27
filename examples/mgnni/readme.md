# Graph Convolutional Networks (GCN)

- Paper link: [https://arxiv.org/abs/2210.08353](https://arxiv.org/abs/2210.08353)
- Author's code repo: [https://github.com/liu-jc/MGNNI](https://github.com/liu-jc/MGNNI). Note that the original code is 
  implemented with Pytorch for the paper. 

# Dataset Statics

| Dataset   | # Nodes | # Edges | # Classes |
|-----------|---------|---------|-----------|
| Cornell   | 183     | 280     | 5         |
| Texas     | 183     | 295     | 5         |
| Wisconsin | 251     | 466     | 5         |

Refer to [WebKB](https://gammagl.readthedocs.io/en/latest/generated/gammagl.datasets.WebKB.html#gammagl.datasets.WebKB).

Results
-------

```bash
# available dataset: "cora", "citeseer", "pubmed"
TL_BACKEND="paddle" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.9 
TL_BACKEND="paddle" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="paddle" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.005 --drop_rate 0.6 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.005 --l2_coef 0.01 --drop_rate 0.6 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.001 --drop_rate 0.8 
TL_BACKEND="tensorflow" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.001 --drop_rate 0.9 
TL_BACKEND="torch" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.005 --l2_coef 0.01 --drop_rate 0.8 
TL_BACKEND="torch" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.7 
TL_BACKEND="torch" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.002 --drop_rate 0.5 
TL_BACKEND="mindspore" python gcn_trainer.py --dataset cora --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.6
TL_BACKEND="mindspore" python gcn_trainer.py --dataset citeseer --num_layers 2 --lr 0.01 --l2_coef 0.05 --drop_rate 0.7 
TL_BACKEND="mindspore" python gcn_trainer.py --dataset pubmed --num_layers 2 --lr 0.01 --l2_coef 0.01 --drop_rate 0.6 
```

| Dataset  | Paper | Our(pd)    | Our(tf)    | Our(th)    | Our(ms)    |
|----------|-------|------------|------------|------------|------------|
| cora     | 81.5  | 81.83±0.22 | 80.54±1.12 | 81.43±0.17 | 81.50±0.64 |
| citeseer | 70.3  | 70.38±0.78 | 68.34±0.68 | 70.53±0.18 | 71.56±0.14 |
| pubmed   | 79.0  | 78.62±0.30 | 78.28±1.08 | 78.63±0.12 | 79.28±0.17 |
