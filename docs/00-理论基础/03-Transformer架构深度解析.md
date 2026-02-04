# 第三章：Transformer架构的深度解析

## 3.1 从序列模型到Transformer

### 3.1.1 RNN的数学瓶颈

RNN的核心递推关系：

$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$$

**问题1：梯度消失**

对 $h_t$ 求 $h_0$ 的梯度：

$$\frac{\partial h_t}{\partial h_0} = \prod_{i=1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

每个雅可比矩阵 $\frac{\partial h_i}{\partial h_{i-1}}$ 的最大奇异值小于1时，梯度指数级衰减。

**问题2：并行化困难**

计算 $h_t$ 必须先计算 $h_{t-1}$，形成严格的时间依赖。

### 3.1.2 注意力如何解决这些问题

注意力机制的优势：

1. **直接建模任意位置依赖**：$\text{Attention}(h_i, H, H)$ 不需要中间传递
2. **完全并行化**：所有位置同时计算

**核心洞察**：Transformer用注意力替代了序列传递。

## 3.2 Transformer架构详解

### 3.2.1 位置编码的数学原理

由于Transformer没有循环结构，需要**显式注入位置信息**。

**正弦位置编码**（原论文）：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**为什么正弦函数有效？**

1. **相对位置**：

$$\cos(a-b) = \cos a \cos b + \sin a \sin b$$

因此 $PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数。

2. **唯一性**：每个位置有唯一的编码
3. **有界性**：编码值在 [-1, 1] 之间

**可学习位置编码**：

将位置编码作为可学习的参数：

$$PE_{pos} = \text{Embedding}(pos)$$

### 3.2.2 编码器结构

每个编码器层包含两个子层：

1. **多头自注意力**（Multi-Head Self-Attention）
2. **前馈网络**（Position-wise Feed-Forward Network）

**残差连接和层归一化**：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**为什么需要残差连接？**

假设子层的理想输出是 $F(x)$，但实际学到的是 $G(x)$。

有残差时：$x + G(x) \approx x + F(x)$（即使$G$有偏差）

没有残差时：$G(x)$ 必须完美逼近 $F(x)$

**层归一化**：

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$

**为什么用LayerNorm而不是BatchNorm？**

- 序列长度可能变化
- BatchNorm依赖batch统计量，不稳定

### 3.2.3 前馈网络

**公式**：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**等价于两次线性变换 + ReLU激活**。

**维度变化**：$d_{model} \to d_{ff} \to d_{model}$

**为什么需要FFN？**

1. **增加非线性**：注意力是线性的（矩阵乘法），需要FFN引入非线性
2. **特征变换**：将注意力输出变换到新空间
3. **容量增加**：FFN参数量是注意力的4倍（$4d^2$）

### 3.2.4 解码器结构

解码器有三层：

1. **掩码自注意力**：防止看到未来位置
2. **交叉注意力**：关注编码器输出
3. **前馈网络**：与编码器相同

**掩码注意力**的作用：

在自注意力中，位置 $i$ 不应该关注位置 $>i$ 的token。

通过掩码矩阵实现：

$$Attention(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M_{ij} = -\infty$ 当 $j > i$，否则为0。

## 3.3 Transformer的计算复杂度

### 3.3.1 各层复杂度分析

| 层 | 时间复杂度 | 空间复杂度 | 并行度 |
|---|----------|-----------|-------|
| 注意力 | $O(n^2 \cdot d)$ | $O(n^2)$ | 高 |
| FFN | $O(n \cdot d^2)$ | $O(n \cdot d)$ | 高 |
| 总计 | $O(n^2 \cdot d)$ | $O(n^2)$ | - |

**当 $n \gg d$ 时**，注意力占主导。

### 3.3.2 序列长度的限制

| 序列长度 | 注意力矩阵 | FFN | 主导 |
|---------|----------|-----|------|
| 512 | 262K | 4M | FFN |
| 2048 | 16M | 16M | 相当 |
| 8192 | 268M | 64M | 注意力 |

这就是为什么需要Longformer、Performer等变体。

### 3.3.3 与RNN对比

| 模型 | 时间复杂度 | 并行度 | 长距离依赖 |
|-----|----------|-------|-----------|
| RNN | $O(n \cdot d^2)$ | 低 | 差 |
| LSTM | $O(n \cdot d^2)$ | 低 | 中 |
| Transformer | $O(n^2 \cdot d)$ | 高 | 强 |

## 3.4 Transformer的归纳偏置

### 3.4.1 什么是归纳偏置

归纳偏置是学习算法对解空间的先验假设。

**RNN的归纳偏置**：
- 序列性：数据是时序生成的
- 局部性：当前位置主要依赖邻近位置

**Transformer的归纳偏置**：
- 任意位置可以直接交互（无偏）
- 需要更多数据和计算来学习位置信息

### 3.4.2 归纳偏置的权衡

**强归纳偏置**（RNN）：
- 优点：小数据量也能学习
- 缺点：难以学习长距离依赖

**弱归纳偏置**（Transformer）：
- 优点：灵活性高，表达能力强
- 缺点：需要更多数据

## 3.5 Transformer的变体

### 3.5.1 编码器-only：BERT

**训练目标**：
1. 掩码语言建模（MLM）：随机掩盖15%的token，预测被掩盖的
2. 下一句预测（NSP）：预测两句话是否连续

**预训练-微调范式**：
- 预训练：在大规模语料上学习通用表示
- 微调：在下游任务上端到端训练

### 3.5.2 解码器-only：GPT

**训练目标**：标准的自回归语言建模

$$L = -\sum_t \log P(w_t | w_1, ..., w_{t-1})$$

**从GPT-1到GPT-4的演进**：
- GPT-1：无监督预训练 + 有监督微调
- GPT-2：零样本提示（展示涌现能力）
- GPT-3：少样本提示（175B参数，涌现In-Context Learning）
- GPT-4：多模态 + RLHF

### 3.5.3 编码器-解码器：T5

**统一文本到文本框架**：
- 翻译：translate English to German: [English] -> [German]
- 摘要：summarize: [Article] -> [Summary]
- 问答：question: [Context] answer: [Answer]

**Span Corruption**：掩盖连续一段span，预测被掩盖的内容

### 3.5.4 混合专家：MoE

**核心思想**：不是所有参数都激活

$$\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i E_i(x)$$

其中 $E_i$ 是第i个专家，$G$ 是路由函数。

**优势**：增加参数量的同时保持计算量不变

## 3.6 本章小结

本章深入解析了Transformer架构：

| 组件 | 公式/机制 | 作用 |
|-----|---------|------|
| 位置编码 | $PE_{(pos,2i)}$ | 注入位置信息 |
| 多头注意力 | $\text{Concat}(heads)W^O$ | 多模式依赖 |
| FFN | $\max(0, xW_1)W_2$ | 非线性变换 |
| 残差连接 | $x + \text{Sublayer}(x)$ | 梯度传递 |
| 层归一化 | $\frac{x-\mu}{\sigma}$ | 稳定训练 |

**核心洞察**：
1. Transformer用注意力替代了序列传递
2. 位置编码是必要的（模型本身无位置概念）
3. 残差连接解决了深层网络的梯度问题

## 参考文献

1. Vaswani, A., et al. (2017). Attention Is All You Need.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
3. Radford, A., et al. (2019). Language Models are Few-Shot Learners.
4. Raffel, C., et al. (2019). Exploring the Limits of Transfer Learning.
5. Fedus, W., et al. (2022). Switch Transformers.
