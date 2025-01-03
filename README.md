# Graph Convolution Network based Recommender Systems_Learning Guarantee and Item Mixture Powered Strategy_ReChorus-fwy_wdb

## 这是一个作业代码
[论文地址](https://openreview.net/forum?id=aUoCgjJfmY9)，在[ReChorus](https://github.com/THUwangcy/ReChorus)框架进行复现

## 环境依赖
将requirements.txt中的numpy版本改为1.23.5后
在  /src 目录下运行
`pip install -r requirements.txt`

## 说明
选用默认数据集 Grocery_and_Gourmet_Food 与 MovieLens_1M
选用基础模型 LightGCN 和 BPRMF，编写了 LightGCNIMix 、 BPRMFIMix

## 运行
输入下面代码以在ReChorus框架下自带的数据集训练
- `python main.py --model_name LightGCN --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M`

------------
--model_name 可以为 LightGCN、BPRMF、LightGCNIMix、BPRMFIMix；--dataset 可以为 Grocery_and_Gourmet_Food 和 MovieLens_1M
也即跑 8 次得到实验结果
