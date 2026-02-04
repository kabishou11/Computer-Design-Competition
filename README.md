# 计算机设计大赛 - 人工智能赛道（大模型方向）指导项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Datawhale](https://img.shields.io/badge/Datawhale-开源学习-blue)](https://github.com/datawhalechina)
[![Stars](https://img.shields.io/github/stars/kabishou11/Computer-Design-Competition)](https://github.com/kabishou11/Computer-Design-Competition/stargazers)
[![Forks](https://img.shields.io/github/forks/kabishou11/Computer-Design-Competition)](https://github.com/kabishou11/Computer-Design-Competition/network/members)

## 项目简介

本项目旨在为参加计算机设计大赛（人工智能赛道）的学生提供系统性的指导，专注于**大模型方向**的理论学习和项目实践。

本项目定位为**大模型理论教程**，而非简单的代码运行指南。内容涵盖从理论基础到实践应用的完整知识体系。

**整合的优秀开源项目**：
- **happy-llm** - LLM系统性学习教程
- **hello-agents** - Agent开发入门指南
- **self-llm** - 本地化部署实践
- **all-in-rag** - RAG技术全栈教程

---

## 学习路径概览

```
人工智能大模型学习路径

├── 第一部分：理论基础（核心）
│   ├── 数学基础
│   ├── 注意力机制原理
│   ├── Transformer架构
│   └── 从词向量到LLM的演进
│
├── 第二部分：技术专题
│   ├── RAG检索增强生成
│   ├── Agent智能体开发
│   ├── 模型微调技术
│   └── 多模态入门
│
├── 第三部分：发展历史
│   ├── NLP发展史
│   └── 经典论文精读
│
├── 第四部分：实践理论
│   ├── 提示工程理论
│   ├── RAG系统框架
│   └── 缩放定律
│
└── 第五部分：项目实战
    └── 完整案例分析
```

---

## 完整目录结构

```
Computer-Design-Competition/
├── docs/                              # 教程文档（40+篇深度文档）
│   │
│   ├── 00-理论基础/                   # 理论核心（新增）
│   │   ├── 01-语言模型的数学基础.md
│   │   ├── 02-注意力机制的数学原理.md
│   │   ├── 03-Transformer架构深度解析.md
│   │   └── 04-从词向量到大语言模型.md
│   │
│   ├── 01-LLM基础/                    # LLM基础知识（4篇）
│   │   ├── 01-NLP基础回顾.md
│   │   ├── 02-Transformer架构.md
│   │   ├── 03-预训练语言模型.md
│   │   └── 04-LLM训练策略.md
│   │
│   ├── 02-RAG技术/                    # RAG检索增强生成（5篇）
│   │   ├── 01-RAG基础概念.md
│   │   ├── 02-文本分块与向量化.md
│   │   ├── 03-向量数据库.md
│   │   ├── 04-混合检索技术.md
│   │   └── 05-高级RAG技术.md
│   │
│   ├── 03-Agent开发/                  # Agent智能体开发（4篇）
│   │   ├── 01-Agent基础概念.md
│   │   ├── 02-ReAct范式.md
│   │   ├── 03-Agent框架实践.md
│   │   └── 04-高级Agent技术.md
│   │
│   ├── 04-模型部署/                   # 模型部署与微调（2篇）
│   │   ├── 01-本地模型部署.md
│   │   └── 02-高效微调技术.md
│   │
│   ├── 05-多模态/                    # 多模态模块（1篇）
│   │   └── 01-多模态大模型入门.md
│   │
│   ├── 06-实战项目/                   # 综合实战项目（3篇）
│   │   ├── 01-智能客服系统.md
│   │   ├── 02-研究助手Agent.md
│   │   └── 03-知识库问答系统.md
│   │
│   ├── 07-发展历史/                   # 发展历史（新增）
│   │   └── 01-自然语言处理发展史.md
│   │
│   ├── 08-论文精读/                   # 论文精读（新增）
│   │   └── 01-Attention-Is-All-You-Need.md
│   │
│   └── 09-实践理论/                   # 实践理论（新增）
│       ├── 01-提示工程的理论原理.md
│       └── 02-RAG系统的理论框架.md
│
├── code/                              # 代码示例
│   ├── RAG/
│   │   ├── basic_rag.py
│   │   └── hybrid_retrieval.py
│   │
│   ├── Agent/
│   │   └── react_agent.py
│   │
│   └── Deployment/
│       ├── local_inference.py
│       ├── api_server.py
│       └── fine_tuning_lora.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 核心理论章节（重点推荐）

### 00-理论基础篇

本部分是项目的**核心价值**，深入讲解大模型背后的数学原理和理论：

| 章节 | 核心内容 |
|-----|---------|
| **语言模型的数学基础** | 概率论基础、N-gram模型、词向量原理、信息论基础 |
| **注意力机制的数学原理** | 缩放点积注意力、多头注意力的数学推导、注意力变体 |
| **Transformer架构深度解析** | 位置编码原理、残差连接与层归一化、计算复杂度分析 |
| **从词向量到LLM** | Word2Vec、ELMo、BERT、GPT的演进脉络 |

### 07-发展历史篇

梳理NLP和大模型的发展脉络：

| 章节 | 核心内容 |
|-----|---------|
| **自然语言处理发展史** | 从规则系统到深度学习、从Word2Vec到Transformer、从BERT到GPT-4 |

### 08-论文精读篇

深度解读经典论文：

| 章节 | 核心内容 |
|-----|---------|
| **Attention Is All You Need** | 论文逐段解析、关键创新分析、实验设计解读 |

### 09-实践理论篇

理论指导实践：

| 章节 | 核心内容 |
|-----|---------|
| **提示工程的理论原理** | 上下文学习理论、提示策略的数学原理 |
| **RAG系统的理论框架** | 检索质量理论、生成条件化分析 |

---

## 特色内容

### 理论深度

每个章节都包含：
- **数学推导**：公式背后的直觉理解
- **历史脉络**：技术演进的内在逻辑
- **关键洞察**：核心原理的深度解读
- **参考文献**：深入学习的权威资源

### 完整知识体系

从**数学基础**到**前沿技术**，形成完整的知识网络：

```
数学基础 ──► 注意力机制 ──► Transformer ──► LLM ──► RAG/Agent
    │              │              │            │
    ▼              ▼              ▼            ▼
概率论      缩放点积      位置编码     GPT/BERT   应用系统
信息论      多头机制      编解码器     缩放定律    评估方法
```

---

## 理论框架概览

### 1. 语言模型的理论基础

```
语言模型 = 概率分布 + 序列建模 + 参数学习

核心问题：如何预测下一个词？

方法演进：
├── N-gram（统计方法）
├── 神经网络语言模型（NNLM）
├── Word2Vec（分布式表示）
├── ELMo（上下文词向量）
├── BERT（双向预训练）
└── GPT系列（自回归预训练）
```

### 2. 注意力机制理论

```
注意力 = 软寻址 + 动态权重 + 加权求和

关键洞察：
├── 查询-键-值三元组
├── 缩放点积的数学原理
└── 多头的表达能力

变体演进：
├── 软注意力
├── 硬注意力
├── 自注意力
└── 稀疏注意力
```

### 3. Transformer架构理论

```
Transformer = 注意力 + 残差 + 层归一化 + FFN

位置编码：
├── 正弦位置编码（绝对位置）
├── 可学习位置编码
└── 相对位置编码（RoPE, ALiBi）

架构演进：
├── 编码器-only（BERT）
├── 解码器-only（GPT）
└── 编码器-解码器（T5）
```

### 4. 大模型缩放定律

```
性能 ∝ 规模^(-α)

关键发现：
├── 计算最优训练（Chinchilla）
├── 涌现能力（Emergent Abilities）
└── 临界点理论

实践指导：
├── 模型规模选择
├── 数据量配置
└── 资源受限策略
```

---

## 技术栈概览

### 核心框架

| 技术领域 | 框架/工具 | 用途 |
|----------|----------|------|
| **LLM框架** | PyTorch, Transformers | 模型训练与推理 |
| **RAG框架** | LangChain, LlamaIndex | 检索增强生成 |
| **Agent框架** | LangGraph, AgentScope | 智能体开发 |
| **向量数据库** | Milvus, FAISS, Chroma | 向量存储与检索 |
| **知识图谱** | Neo4j | 知识图谱存储 |
| **低代码平台** | Dify, Coze | 快速原型开发 |
| **多模态** | LLaVA, BLIP-2 | 视觉语言模型 |
| **评估工具** | RAGAs | RAG系统评估 |

### 模型选择建议

根据硬件条件和项目需求选择合适的模型：

| 场景 | 推荐模型 | 显存要求 | 说明 |
|------|----------|----------|------|
| **入门学习** | Qwen3-1.5B, Phi-3.5-mini | 4-8GB | 适合本地CPU/GPU推理 |
| **应用开发** | Qwen3-7B, Llama3-8B | 16-24GB | 平衡性能与资源 |
| **竞赛项目** | Qwen3-14B, Yi-1.5-34B | 48GB+ | 高性能需求 |
| **API调用** | GPT-4, Claude-3.5, 通义千问 | 无需本地 | 云端服务 |

---

## 比赛选题建议（例子，仅作学习参考）

### 第一阶段：理论基础（3-4周）

1. **第1周**：语言模型的数学基础
2. **第2周**：注意力机制原理
3. **第3周**：Transformer架构解析
4. **第4周**：从词向量到LLM的演进

### 第二阶段：技术专题（4-6周）

1. **RAG技术**（2周）
2. **Agent开发**（2周）
3. **微调技术**（1周）
4. **多模态**（1周）

### 第三阶段：论文研读（2周）

1. 精读《Attention Is All You Need》
2. 了解前沿研究方向

### 第四阶段：项目实践（4周）

1. 选择项目方向
2. 设计与实现
3. 评估与优化

---

## 项目特色

### 理论深度

- 每个概念都有数学推导
- 每个技术都有历史脉络
- 每个架构都有原理解析

### 知识完整

- 从基础到前沿
- 从理论到实践
- 从历史到未来

### 学习导向

| 项目 | 技术栈 | 天数 | 目标 |
|-----|--------|-----|------|
| 智能客服系统 | RAG + 对话管理 | 1周 | 完整可运行系统 |
| 知识库问答 | 向量检索 + 重排序 | 1周 | 高质量检索 |
| 研究助手Agent | Agent + 多工具 | 2周 | 多功能集成 |

### 第四阶段：比赛准备（建议2周）

| 任务 | 内容 | 天数 |
|-----|------|-----|
| 文档整理 | 技术报告、PPT | 5天 |
| 答辩演练 | 模拟答辩、Q&A准备 | 5天 |

---

## 环境配置

### 基础环境要求

- **Python**: 3.10+
- **CUDA**: 12.0+（GPU训练需要）
- **RAM**: 16GB+
- **硬盘**: 50GB+

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/kabishou11/Computer-Design-Competition.git
cd Computer-Design-Competition

# 安装依赖
pip install -r requirements.txt

# 或分步安装
pip install torch transformers accelerate

# RAG相关
pip install langchain llama-index sentence-transformers pymilvus faiss-cpu chromadb

# Agent相关
pip install langgraph agentscope

# 评估工具
pip install ragas
```

### Docker部署（可选）

```bash
# 使用Docker运行
docker build -t llm-course .
docker run -p 8000:8000 llm-course
```

---

## 快速开始

### 1. 学习路线

```bash
# 第一步：阅读基础知识
docs/01-LLM基础/

# 第二步：学习RAG技术
docs/02-RAG技术/

# 第三步：掌握Agent开发
docs/03-Agent开发/

# 第四步：实践项目
docs/05-实战项目/
```

### 2. 运行示例代码

```bash
# RAG示例
python code/RAG/basic_rag.py

# Agent示例
python code/Agent/react_agent.py

# 部署示例
python code/Deployment/api_server.py
```

---

## 新增内容（2024-2026最新技术）

### 高级RAG技术

- 查询优化（HyDE、查询改写）
- Agentic RAG（Agent驱动检索）
- Self-RAG（自反思RAG）
- RAGAs评估方法

### 高级Agent技术

- 记忆系统（短期+长期）
- 多Agent协作（层级、辩论）
- 自定义工具构建
- Agent评估框架

### 多模态入门

- LLaVA视觉语言模型
- BLIP-2图像理解
- 语音识别与合成
- 多模态应用场景
>>>>>>> e64a95ec3526df3c80b52a3dc59260538d8d053b

---

## 贡献指南

本项目需要理论贡献：

1. **理论补充**：添加新的理论章节
2. **论文解读**：精读更多经典论文
3. **案例分析**：深度分析优秀项目
4. **文献整理**：整理相关参考文献

---

## 许可证

MIT License

---

## 致谢

感谢以下开源项目和社区：

- [Datawhale](https://github.com/datawhalechina) - 优质学习资源
- [Hugging Face](https://huggingface.co/) - Transformers库
- [LangChain AI](https://github.com/langchain-ai) - Agent和RAG框架

---

<div align="center">

**本项目是一本大模型理论教程，帮助你深入理解大模型背后的原理。**

**如果对你有帮助，请点个Star支持一下！**

</div>
