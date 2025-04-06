# BackdoorShield

![PyTorch Version](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)

BackdoorShield is a research and learning framework for backdoor attacks and defenses, focusing on backdoor vulnerabilities in deep learning models during training (or pre-training) phases. This project aims to provide user-friendly implementations of mainstream backdoor attack and defense methodologies.（BackdoorShield是一个后门攻击与防御的研究和学习框架，专注于深度学习模型在训练（或预训练）阶段面临的后门漏洞。该项目旨在为主流的常用的后门攻击与防御方法提供易用的实现方案。）

### 攻击方法:

| 方法 | 文件名 | 论文 | 基本思想 |
|---------|---------|---------|---------|
| SOS | [SOS.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/SOS.py "点击访问 ") | Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun: Rethinking Stealthiness of Backdoor Attack against NLP Models. ACL/IJCNLP (1) 2021: 5543-5557 [[Paper]](https://aclanthology.org/2021.acl-long.431 "paper")                                 |提出新型隐蔽后门攻击方法（SOS框架）。后门仅当所有预定义的触发词（例如“friend”、“cinema”、“weekend”三个词）同时出现时才激活。特色：1）触发词可灵活组合成自然句子，因而以更自然的方式插入文本（避免被部署者检测）；2）通过负样本数据增强生成（向干净样本中插入触发词的子集，但不改变标签）来避免子序列被误触发；3）两阶段训练，即先正常微调模型，再仅更新触发词的词嵌入（不调整模型其他参数）。|
| 单元格4 | 单元格5 | 单元格6 |单元格6 |
| 单元格4 | 单元格5 | 单元格6 |单元格6 |
| 单元格4 | 单元格5 | 单元格6 |单元格6 |

### 防御方法:

| 方法 | 文件名 | 论文 | 基本思想 |
|---------|---------|---------|---------|
| 单元格4 | 单元格5 | 单元格6 |单元格6 |

### Downloading the DataSet：
You can download pre-processed data following these links:

[Spam](https://github.com/neulab/RIPPLe/releases/download/data/spam_data.zip "Spam") 包括： enron，lingspam 两种数据集

[Sentiment](https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip "Sentiment") 包括：amazon, imdb, SST-2, yelp 四种数据集。

[Toxicity](https://github.com/neulab/RIPPLe/releases/download/data/toxic_data.zip "Toxicity") 包括：jigsaw，offenseva，Twitter 三种数据集。


## 攻击（数据毒化）和防御（ONION）如下:

**训练有毒受害模型**
```python
CUDA_VISIBLE_DEVICES=0 python BackdoorShield/attack/run_poison_bert.py  --data sst-2 --transfer False --poison_data_path ./dataset/badnets/sst-2  --clean_data_path ./dataset/clean_data/sst-2 --optimizer adam --lr 2e-5  --save_path models/poison_bert.pkl
```

**用ONION防御**

```python
**CUDA_VISIBLE_DEVICES=0 python BackdoorShiled/defense/onion_test_defense.py  --data sst-2 --model_path models/poison_bert.pkl  --poison_data_path ./dataset/badnets/sst-2/test.tsv  --clean_data_path ./dataset/clean_data/sst-2/dev.tsv**
```

**防御效果可视化：**
```python
python BackdoorShield/evaluate/eval_onion.py
```

## SOS攻击+RAP防御

**攻击部分**

先划分数据集为两部分
```python
 python3 BackdoorShield/data/split_train_and_dev.py --task sentiment --dataset imdb --split_ratio 0.9
```
训练模型
```python
python3 BackdoorShield/train/clean_model_train.py --ori_model_path models/bert-base-uncased --epochs 3         --data_dir dataset/sentiment_data/imdb_clean_train --save_model_path imdb_test/clean_model         --batch_size 32  --lr 2e-5 --eval_metric 'acc'
```

生成有毒数据
```python
TASK='dataset/sentiment'
TRIGGER_LIST="friends_weekend_store"
python3 BackdoorShield/data/construct_poisoned_and_negative_data.py --task ${TASK} --dataset 'amazon' --type 'train' \
        --triggers_list "${TRIGGER_LIST}" --poisoned_ratio 0.1 --keep_clean_ratio 0.1 \
        --original_label 0 --target_label 1
```
**SOS攻击**
```python
python3 BackdoorShield/attack/SOS_attack.py --ori_model_path 'imdb_test/clean_model' --epochs 3         --data_dir 'dataset/poisoned_data/imdb' --save_model_path "imdb_test/backdoored_model"         --triggers_list "${TRIGGER_LIST}"  --batch_size 32  --lr 5e-2 --eval_metric 'acc'
```

**RAP防御**
```python
python3 BackdoorShield/defense/rap_defense.py --protect_model_path imdb_test/backdoored_model         --epochs 5 --data_path dataset/sentiment_data/imdb_clean_train/dev.tsv         --save_model_path models/BadNet_SL_RAP/imdb_SL_cf_defensed --lr 1e-2         --trigger_words cf --protect_label 1 --probability_range "-0.1 -0.3"         --scale_factor 1 --batch_size 16
```

**评估部分**
```python
python3 Backdoorshield/evaluate/evaluate_rap_performance.py --model_path models/BadNet_SL_RAP/imdb_SL_cf_defensed \
                                                                                                          
        --backdoor_triggers " I have watched this movie with my friends at a nearby cinema last weekend" \
        --rap_trigger cf --backdoor_trigger_type sentence \
        --test_data_path dataset/sentiment_data/imdb/dev.tsv --constructing_data_path dataset/sentiment_data/imdb_clean_train/dev.tsv \
        --batch_size 64 --protect_label 1
```
