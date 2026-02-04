# 第二章：Transformer架构详解

## 2.1 Transformer的诞生

2017年，Google在论文《Attention Is All You Need》中提出了Transformer架构，完全基于注意力机制，摒弃了循环和卷积结构。这一创新彻底改变了自然语言处理领域，为后续GPT、BERT等大语言模型奠定了基础。

### 2.1.1 Transformer vs 传统序列模型

| 特性 | RNN/LSTM | Transformer |
|-----|----------|-------------|
| **并行计算** | 难以并行 | 完全并行 |
| **长程依赖** | 易梯度消失 | 有效捕捉 |
| **位置信息** | 天然包含 | 需额外编码 |
| **计算复杂度** | O(n) | O(n²) |

### 2.1.2 Transformer整体架构

```
输入嵌入
    │
    ▼
┌─────────────────────────────┐
│       Positional Encoding   │  ← 位置编码
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│      Encoder Layer × N      │  ← 编码器层
│  ├── Multi-Head Attention   │
│  ├── Add & Norm             │
│  ├── Feed Forward           │
│  └── Add & Norm             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│      Decoder Layer × N      │  ← 解码器层
│  ├── Masked Multi-Head      │
│  ├── Add & Norm             │
│  ├── Cross Attention         │
│  ├── Add & Norm             │
│  ├── Feed Forward           │
│  └── Add & Norm             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│         Linear + Softmax    │  ← 输出层
└─────────────────────────────┘
    │
    ▼
        输出概率
```

## 2.2 位置编码（Positional Encoding）

由于Transformer没有循环结构，需要额外注入位置信息。

### 2.2.1 正弦位置编码

```python
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 使用示例
d_model = 512
pos_encoder = PositionalEncoding(d_model)
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
encoded = pos_encoder(x)
print("位置编码后形状:", encoded.shape)
```

### 2.2.2 可学习位置编码

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand_as(x)
        position_embeddings = self.position_embeddings(positions)
        return self.LayerNorm(x + position_embeddings)
```

## 2.3 多头注意力机制（Multi-Head Attention）

### 2.3.1 核心原理

多头注意力将输入分成多个"头"，每个头学习不同的表示子空间：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax + Dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 加权求和
        out = torch.matmul(attention, V)

        # 拼接多个头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(out)

        return out, attention
```

### 2.3.2 使用示例

```python
# 参数设置
d_model = 512
num_heads = 8
batch_size = 32
seq_len = 100

# 创建注意力层
mha = MultiHeadAttention(d_model, num_heads)

# 示例输入
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output, attention_weights = mha(Q, K, K)
print("输出形状:", output.shape)  # (32, 100, 512)
print("注意力权重形状:", attention_weights.shape)  # (32, 8, 100, 100)
```

## 2.4 前馈神经网络（Feed Forward Network）

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```

## 2.5 残差连接与层归一化

```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

## 2.6 完整的Encoder层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

## 2.7 完整的Decoder层

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 自注意力（带掩码）
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 交叉注意力
        x = self.residual2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        # 前馈网络
        x = self.residual3(x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```

## 2.8 完整的Transformer模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        # 编码
        src_embedded = self.pos_encoder(self.src_embed(src))
        encoder_output = self.encoder(src_embedded, src_mask)

        # 解码
        tgt_embedded = self.pos_encoder(self.tgt_embed(tgt))
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        return self.generator(decoder_output)

# 使用示例
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

src = torch.randint(0, 10000, (32, 100))
tgt = torch.randint(0, 10000, (32, 50))
output = model(src, tgt)
print("输出形状:", output.shape)  # (32, 50, 10000)
```

## 2.9 Transformer的变体

| 变体 | 特点 | 代表模型 |
|-----|------|---------|
| **Encoder-only** | 双向注意力，适合理解任务 | BERT, RoBERTa |
| **Decoder-only** | 单向注意力，适合生成任务 | GPT, LLaMA |
| **Encoder-Decoder** | 完整结构，适合翻译任务 | T5, BART |

## 2.10 本章小结

本章详细介绍了Transformer的核心组件：
- 位置编码（注入序列位置信息）
- 多头注意力（并行学习多表示空间）
- 前馈神经网络（非线性变换）
- 残差连接与层归一化（稳定训练）
- Encoder-Decoder架构

理解这些基础组件对于后续学习GPT、BERT等大语言模型至关重要。

## 练习题

1. 实现一个简单的Transformer编码器层
2. 比较不同位置编码方法的优缺点
3. 分析多头注意力中"头数"对模型性能的影响
4. 思考：为什么Transformer比RNN更适合GPU并行计算？

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解Transformer
- [Transformer Anatomy](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP实现
