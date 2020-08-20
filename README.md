# 小说生成模型

应用one-hot形式，在不带入预训练embedding的情况下进行文本生成。

## 环境

Python 3+

Tensorflow 2+

## 语料

放置于'./data'中，使用《白夜行》第一章作为训练集

## 模型

目前尝试BidirectionalGRU的形式进行生成，主要就是对于输入的数据预测下一个词会是什么

## 运行

使用'./src/main.py'进行训练以及测试



