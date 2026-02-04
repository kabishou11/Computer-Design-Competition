# 第一章：NLP基础回顾

## 1.1 什么是自然语言处理

自然语言处理（Natural Language Processing, NLP）是人工智能的一个重要分支，致力于让计算机能够理解、解释和生成人类语言。从早期的规则系统到现在的深度学习方法，NLP经历了巨大的发展变革。

### NLP的主要任务

| 任务类型 | 具体任务 | 应用场景 |
|---------|---------|---------|
| **文本理解** | 文本分类、情感分析、命名实体识别 | 舆情监控、信息抽取 |
| **文本生成** | 机器翻译、摘要生成、文本补全 | 写作辅助、内容创作 |
| **问答系统** | 阅读理解、开放域问答 | 智能客服、知识问答 |
| **信息检索** | 文本匹配、关键词提取 | 搜索引擎、推荐系统 |

## 1.2 文本表示方法

### 1.2.1 词袋模型（Bag of Words）

最简单的文本表示方法，将文本表示为词频向量：

```python
from sklearn.feature_extraction.text import CountVectorizer

# 示例文档
documents = [
    "人工智能改变世界",
    "深度学习是AI的重要分支",
    "NLP让机器理解人类语言"
]

# 创建词袋模型
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print("词汇表:", vectorizer.get_feature_names_out())
print("词袋向量:\n", bow_matrix.toarray())
```

### 1.2.2 TF-IDF

TF-IDF（词频-逆文档频率）是一种更精细的文本表示方法：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("TF-IDF向量:\n", tfidf_matrix.toarray())
```

**计算公式：**
- TF（词频）: `TF(t,d) = 词t在文档d中出现次数 / 文档d总词数`
- IDF（逆文档频率）: `IDF(t,D) = log(文档总数 / 包含词t的文档数)`

### 1.2.3 词向量（Word Embedding）

词向量将词语映射到稠密的低维向量空间，使语义相近的词在向量空间中距离更近：

```python
from gensim.models import Word2Vec
import numpy as np

# 训练词向量模型
sentences = [
    ["人工", "智能", "是", "未来"],
    ["深度", "学习", "需要", "数据"],
    ["模型", "训练", "很", "重要"]
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)

# 获取词向量
word_vector = model.wv["人工"]
print("'人工'的词向量:", word_vector)

# 词语相似度
similarity = model.wv.similarity("人工", "智能")
print("'人工'和'智能'的相似度:", similarity)
```

## 1.3 序列模型基础

### 1.3.1 RNN（循环神经网络）

RNN是最早用于处理序列数据的神经网络结构：

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 只取最后一个时刻的输出
        return out

# 使用示例
model = SimpleRNN(input_size=50, hidden_size=100, output_size=2)
x = torch.randn(32, 10, 50)  # (batch, seq_len, input_size)
output = model(x)
print("输出形状:", output.shape)
```

**RNN的问题：**
- 梯度消失/爆炸问题
- 长序列信息丢失
- 无法并行计算

### 1.3.2 LSTM（长短期记忆网络）

LSTM通过门控机制解决了RNN的长程依赖问题：

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        output = self.classifier(lstm_out[:, -1, :])
        return output

# 使用示例
vocab_size = 10000
model = LSTMModel(vocab_size, 128, 256)
input_ids = torch.randint(0, vocab_size, (32, 100))  # (batch, seq_len)
output = model(input_ids)
print("输出形状:", output.shape)
```

### 1.3.3 GRU（门控循环单元）

GRU是LSTM的简化版本，参数更少但效果相近：

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = torch.zeros(x.size(0), x.size(1), x.size(2))
        out, _ = self.gru(x, h0)
        return out
```

## 1.4 注意力机制

### 1.4.1 什么是注意力机制

注意力机制（Attention Mechanism）让模型能够动态地关注输入序列的不同部分，是Transformer架构的核心。

```python
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分割嵌入向量为多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)
```

### 1.4.2 注意力类型

| 类型 | 说明 | 应用场景 |
|-----|------|---------|
| **自注意力** | Q、K、V来自同一序列 | Transformer |
| **编码器-解码器注意力** | Q来自解码器，K、V来自编码器 | 机器翻译 |
| **多头注意力** | 多组注意力并行计算 | Transformer |

## 1.5 本章小结

本章回顾了NLP的基础知识，包括：
- 文本表示方法（BoW、TF-IDF、词向量）
- 序列模型（RNN、LSTM、GRU）
- 注意力机制原理

这些知识是学习Transformer和大语言模型的基础。在下一章中，我们将深入学习Transformer架构。

## 练习题

1. 手动实现一个简单的词向量模型
2. 比较RNN和LSTM在长序列上的表现差异
3. 实现一个多头注意力模块
4. 思考：为什么注意力机制比循环结构更适合并行计算？

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) - Stanford NLP教材
- [Neural Network Methods for NLP](https://www.amazon.com/Neural-Network-Methods-Natural-Language/dp/1617795766) - NLP神经网络方法
