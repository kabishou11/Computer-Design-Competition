# 计算机设计大赛 - 人工智能赛道（大模型方向）指导项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Datawhale](https://img.shields.io/badge/Datawhale-开源学习-blue)](https://github.com/datawhalechina)
[![Stars](https://img.shields.io/github/stars/kabishou11/Computer-Design-Competition)](https://github.com/kabishou11/Computer-Design-Competition/stargazers)
[![Forks](https://img.shields.io/github/forks/kabishou11/Computer-Design-Competition)](https://github.com/kabishou11/Computer-Design-Competition/network/members)

## 项目简介

本项目旨在为参加计算机设计大赛（人工智能赛道）的学生提供系统性的指导，专注于**大模型方向**的技术学习和项目实践。

整合了以下优秀开源项目的精华内容：
- **happy-llm** - LLM系统性学习教程
- **hello-agents** - Agent开发入门指南
- **self-llm** - 本地化部署实践
- **all-in-rag** - RAG技术全栈教程

帮助学生从零开始掌握大模型的核心技术栈，并应用于比赛项目。

---

## 学习路径概览

```
人工智能大模型学习路径

├── 第一阶段：基础知识（1-2周）
│   ├── LLM基础概念
│   ├── Transformer架构
│   └── NLP基础回顾
│
├── 第二阶段：核心技术（3-4周）
│   ├── RAG检索增强生成
│   ├── Agent智能体开发
│   ├── 模型部署与微调
│   └── 多模态入门
│
├── 第三阶段：项目实战（4-6周）
│   ├── 智能客服系统
│   ├── 知识库问答系统
│   └── 研究助手Agent
│
└── 第四阶段：比赛准备（2周）
    ├── 选题与规划
    ├── 项目开发
    └── 答辩材料准备
```

---

## 完整目录结构

```
Computer-Design-Competition/
├── docs/                              # 教程文档（25+篇文档）
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
│   │   └── 05-高级RAG技术.md          # [新增]查询优化、Agentic RAG
│   │
│   ├── 03-Agent开发/                  # Agent智能体开发（4篇）
│   │   ├── 01-Agent基础概念.md
│   │   ├── 02-ReAct范式.md
│   │   ├── 03-Agent框架实践.md
│   │   └── 04-高级Agent技术.md        # [新增]记忆系统、多Agent协作
│   │
│   ├── 04-模型部署/                   # 模型部署与微调（2篇）
│   │   ├── 01-本地模型部署.md
│   │   └── 02-高效微调技术.md
│   │
│   ├── 05-多模态/                    # 多模态模块（1篇）
│   │   └── 01-多模态大模型入门.md      # [新增]视觉语言模型
│   │
│   ├── 05-实战项目/                   # 综合实战项目（3篇）
│   │   ├── 01-智能客服系统.md          # [新增]完整系统实现
│   │   ├── 02-研究助手Agent.md         # [新增]学术研究辅助
│   │   └── 03-知识库问答系统.md        # [新增]文档问答系统
│   │
│   └── 06-比赛指南/                   # 比赛相关指导（4篇）
│       ├── 01-比赛选题指南.md
│       ├── 02-项目评估标准.md
│       ├── 03-答辩技巧.md
│       └── 04-常见问题FAQ.md
│
├── code/                              # 代码示例（6个完整项目）
│   ├── RAG/
│   │   ├── basic_rag.py               # 基础RAG实现
│   │   └── hybrid_retrieval.py         # 混合检索实现
│   │
│   ├── Agent/
│   │   └── react_agent.py             # ReAct Agent实现
│   │
│   └── Deployment/
│       ├── local_inference.py          # 本地模型推理
│       ├── api_server.py               # FastAPI服务
│       └── fine_tuning_lora.py         # LoRA微调示例
│
├── data/                              # 示例数据
│   ├── knowledge_base/                 # 知识库样例
│   └── datasets/                      # 数据集
│
├── requirements.txt                    # 依赖清单
├── LICENSE
└── README.md                          # 项目说明
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
| **入门学习** | Qwen2.5-1.5B, Phi-3.5-mini | 4-8GB | 适合本地CPU/GPU推理 |
| **应用开发** | Qwen2.5-7B, Llama3-8B | 16-24GB | 平衡性能与资源 |
| **竞赛项目** | Qwen2.5-14B, Yi-1.5-34B | 48GB+ | 高性能需求 |
| **API调用** | GPT-4, Claude-3.5, 通义千问 | 无需本地 | 云端服务 |

---

## 比赛选题建议

### 应用类项目推荐（适合资金有限）

以下项目类型投资小、见效快，非常适合学校老师资金不支持的情况：

| 项目类型 | 技术要点 | 创新点建议 | 难度 | 推荐度 |
|----------|----------|-----------|------|-------|
| **智能客服** | RAG + 对话管理 | 垂直领域知识库 | ⭐⭐ | 高 |
| **知识问答系统** | RAG + 向量检索 | 特定领域知识图谱 | ⭐⭐ | 高 |
| **文档处理助手** | OCR + LLM + Agent | 自动化文档工作流 | ⭐⭐⭐ | 中 |
| **研究助手** | Agent + 搜索增强 | 多源信息整合 | ⭐⭐⭐ | 高 |
| **创意写作工具** | Prompt工程 + 微调 | 特定风格生成 | ⭐⭐ | 中 |

### 不推荐的挑战类项目

以下项目需要大量计算资源，不建议在资金有限的情况下尝试：

- 从零训练大语言模型
- 超大规模分布式训练
- 多模态大模型训练
- 复杂强化学习训练

### 优秀选题方向

1. **教育领域**
   - 智能答疑系统
   - 个性化学习助手
   - 作业批改助手

2. **医疗健康**
   - 疾病知识问答
   - 用药助手
   - 健康咨询机器人

3. **企业办公**
   - 会议纪要生成
   - 合同审核助手
   - 知识管理系统

4. **文化创意**
   - 故事生成器
   - 诗歌创作助手
   - 设计灵感工具

---

## 学习路线图（详细版）

### 第一阶段：基础知识（建议2周）

| 章节 | 内容 | 天数 | 重点 |
|-----|------|-----|------|
| NLP基础回顾 | 文本表示、词向量、RNN/LSTM | 3天 | 理解传统NLP方法 |
| Transformer架构 | 注意力机制、位置编码 | 4天 | **核心重点** |
| 预训练语言模型 | BERT/GPT原理、训练目标 | 3天 | 理解预训练范式 |
| LLM训练策略 | SFT、LoRA、RLHF | 4天 | 掌握微调方法 |

### 第二阶段：核心技术（建议4周）

| 章节 | 内容 | 天数 | 重点 |
|-----|------|-----|------|
| RAG基础 | 检索-生成范式 | 3天 | 理解RAG原理 |
| 文本分块与向量化 | 嵌入模型选择 | 3天 | **实践重点** |
| 向量数据库 | Milvus/FAISS/Chroma | 3天 | 掌握至少一种 |
| 混合检索 | BM25、重排序 | 3天 | 提升检索质量 |
| Agent基础 | 工具调用、推理链 | 4天 | 理解Agent范式 |
| ReAct范式 | 思考-行动-观察循环 | 3天 | **核心重点** |

### 第三阶段：项目实战（建议4周）

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

## 新增内容（2024-2025最新技术）

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

---

## 贡献指南

欢迎贡献教程、代码示例和项目案例！

1. **Fork** 本项目
2. **创建** 特性分支 (`git checkout -b feature/AmazingFeature`)
3. **提交** 更改 (`git commit -m 'Add some AmazingFeature'`)
4. **推送** 到分支 (`git push origin feature/AmazingFeature`)
5. 发起 **Pull Request**

### 贡献方向

- 补充更多代码示例
- 添加实战项目案例
- 完善文档内容
- 翻译为其他语言
- 修正错误

---

## 许可证

本项目采用 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

---

## 致谢

感谢以下开源项目和社区：

- [Datawhale](https://github.com/datawhalechina) - 提供优质学习资源
- [Hugging Face](https://huggingface.co/) - Transformers库
- [LangChain AI](https://github.com/langchain-ai) - Agent和RAG框架
- [LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) - 微调工具

---

## 联系方式

- **GitHub**: [kabishou11/Computer-Design-Competition](https://github.com/kabishou11/Computer-Design-Competition)
- **Issues**: 使用GitHub Issues提问
- **Discussions**: 参与社区讨论

---

<div align="center">

**如果本项目对你有帮助，请点个Star支持一下！**

</div>
