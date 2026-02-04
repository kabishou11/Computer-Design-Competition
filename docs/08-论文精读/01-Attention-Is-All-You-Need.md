# 第二章：Attention Is All You Need 论文精读

## 2.1 论文背景

### 2.1.1 研究动机

**问题**：
- RNN在长序列上存在梯度消失问题
- LSTM虽然缓解但仍无法完全解决
- RNN无法并行计算，训练效率低

**已有工作的局限**：
- 注意力机制通常与RNN结合
- 并行化程度有限
- 计算复杂度高

### 2.1.2 核心贡献

本文提出**Transformer**，完全基于注意力机制：

1. **自注意力**（Self-Attention）：序列内部的关系建模
2. **多头注意力**：多种注意力模式的组合
3. **位置编码**：弥补序列顺序信息
4. **编解码器结构**：标准的序列到序列架构

## 2.2 核心架构分析

### 2.2.1 整体架构图解

```
输入嵌入                    N×编码器层                    N×解码器层           输出
   │                            │                          │                │
   ▼                            ▼                          ▼                ▼
┌───────┐               ┌──────────────┐            ┌──────────────┐        ┌───────┐
│词嵌入 │               │ 注意力层     │            │ 注意力层     │        │线性层 │
│+位置 │──────►  ──►│  (多头)      │──►  ──►│  (掩码多头)  │──►  ──►│+softmax│
└───────┘               │  +残差      │            │  +残差       │        └───────┘
                        └──────────────┘            └──────────────┘
                        ┌──────────────┐            ┌──────────────┐
                        │  前馈网络    │            │  交叉注意力  │
                        │  +残差      │            │  +残差       │
                        └──────────────┘            └──────────────┘
```

### 2.2.2 编码器详解

**每个编码器层**：

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 子层1：多头自注意力 + 残差连接
        x1 = self.layer_norm1(x)
        x1 = self.self_attn(x1, x1, x1, mask)
        x = x + self.dropout(x1)

        # 子层2：前馈网络 + 残差连接
        x2 = self.layer_norm2(x)
        x2 = self.feed_forward(x2)
        x = x + self.dropout(x2)

        return x
```

**设计选择**：

1. **残差连接**：帮助梯度传播
2. **层归一化**：稳定训练
3. **子层归一化**：先归一化后注意力（与原版不同）

### 2.2.3 解码器详解

**每个解码器层**：

```python
class DecoderLayer(nn.Module):
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 子层1：掩码多头自注意力
        x1 = self.layer_norm1(x)
        x1 = self.self_attn(x1, x1, x1, tgt_mask)
        x = x + self.dropout(x1)

        # 子层2：交叉注意力
        x2 = self.layer_norm2(x)
        x2 = self.cross_attn(x2, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(x2)

        # 子层3：前馈网络
        x3 = self.layer_norm3(x)
        x3 = self.feed_forward(x3)
        x = x + self.dropout(x3)

        return x
```

**掩码的作用**：

防止位置 $i$ 看到位置 $>i$ 的内容：

```
位置0: [A, M, M, M]  # 只能看到自己
位置1: [A, A, M, M]  # 看到位置0和1
位置2: [A, A, A, M]  # 看到位置0,1,2
位置3: [A, A, A, A]  # 看到所有位置
```

## 2.3 关键技术分析

### 2.3.1 多头注意力

**为什么用多头？**

单头注意力的局限：
- 只能关注一种模式
- 不同头可以关注不同方面

**头的设计**：

$$d_k = d_v = d_{model} / h$$

每个头降维，合并后与单头复杂度相当。

### 2.3.2 位置编码

**正弦位置编码公式**：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**为什么正弦有效？**

1. **相对位置**：
$$\cos(a-b) = \cos a \cos b + \sin a \sin b$$

2. **线性变换**：
$$PE(pos + k)$$ 可以表示为 $PE(pos)$ 的线性组合

### 2.3.3 缩放点积

**公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**为什么缩放？**

点积的期望为0，方差为 $d_k$：
- $d_k$ 大时，点积值大
- softmax梯度极小
- 缩放后方差为1

## 2.4 训练技巧

### 2.4.1 正则化

1. **Dropout**：应用于注意力权重和残差连接
2. **标签平滑**：0.1的标签平滑

### 2.4.2 优化器

**Adam with warmup**：

$$\eta(t) = \eta_{max} \cdot \frac{t}{warmup\_steps}$$

### 2.4.3 训练超参数

| 参数 | 值 |
|-----|-----|
| 层数 | 6 |
| 注意力头数 | 8 |
| $d_{model}$ | 512 |
| $d_{ff}$ | 2048 |
| Dropout | 0.1 |
| Batch Size | 4096 tokens |

## 2.5 实验分析

### 2.5.1 翻译任务

| 模型 | BLEU | 参数量 | 计算量 |
|-----|------|-------|--------|
| Transformer (base) | 27.3 | 65M | 低 |
| Transformer (big) | 28.4 | 210M | 高 |
| ConvS2S | 25.3 | 210M | 高 |
| ByteNet | 23.7 | 95M | 中 |

### 2.5.2 注意力可视化

**论文中的观察**：

1. **自注意力**：学习到语法依赖关系
2. **头专业化**：不同头关注不同方面
3. **长距离依赖**：比RNN更好地捕捉

### 2.5.3 消融实验

移除不同组件的影响：

| 组件 | BLEU变化 |
|-----|---------|
| 完整模型 | 28.4 |
| 移除多头 | -0.9 |
| 移除位置编码 | -2.8 |
| 移除残差 | -2.1 |

## 2.6 深远影响

### 2.6.1 对后续研究的影响

1. **BERT**：只使用编码器
2. **GPT**：只使用解码器
3. **ViT**：应用于视觉
4. **AlphaFold**：结构生物学

### 2.6.2 产业影响

- **Google翻译**：全面切换到Transformer
- **搜索排名**：注意力模型用于理解查询
- **推荐系统**：注意力机制广泛使用

## 2.7 论文批判性分析

### 2.7.1 局限性

1. **$O(n^2)$复杂度**：长序列挑战
2. **位置编码**：相对位置表示不足
3. **解码器自回归**：生成速度慢

### 2.7.2 后续改进

| 改进方向 | 代表工作 | 解决的问题 |
|---------|---------|-----------|
| 高效注意力 | Longformer, Performer | $O(n^2)$复杂度 |
| 相对位置 | Transformer-XL | 位置编码限制 |
| 非自回归 | Non-autoregressive | 生成速度 |
| 位置编码 | RoPE, ALiBi | 更强的位置表示 |

## 2.8 本章小结

**核心洞察**：

1. 注意力机制完全替代RNN是可行的
2. 多头设计增加了模型的表达能力
3. 位置编码是必要的创新
4. 残差连接和层归一化是关键训练技巧

**历史地位**：

Transformer是深度学习时代最重要的论文之一，开启了NLP的新纪元。

## 参考文献

Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
