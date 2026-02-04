# 第一章：RAG基础概念

## 1.1 什么是RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了检索（Retrieval）和生成（Generation）的技术框架。其核心思想是：

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG工作流程                               │
│                                                                 │
│   用户查询                                                       │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────┐                                                 │
│   │  检索模块   │ ───► 从知识库检索相关文档                       │
│   └───────────┘                                                 │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────┐                                                 │
│   │  融合模块   │ ───► 将检索结果与查询融合                      │
│   └───────────┘                                                 │
│       │                                                         │
│       ▼                                                         │
│   ┌───────────┐                                                 │
│   │  生成模块   │ ───► 基于检索结果生成回答                       │
│   └───────────┘                                                 │
│       │                                                         │
│       ▼                                                         │
│     回答                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.1 为什么需要RAG

| 问题 | 传统LLM | RAG |
|-----|--------|-----|
| **知识时效性** | 训练数据截止，无法获取新知识 | 实时检索最新信息 |
| **幻觉问题** | 可能生成错误信息 | 基于检索事实，减少幻觉 |
| **领域知识** | 缺乏特定领域知识 | 可注入专业领域知识 |
| **可控性** | 输出难以控制 | 可通过检索源控制输出 |
| **成本** | 大模型训练成本高 | 增量更新知识库成本低 |

### 1.1.2 RAG的发展历程

```
2020        2021        2022        2023        2024
  │           │           │           │           │
  ●───────────●───────────●───────────●───────────●
Facebook     RAG+        LangChain   高级RAG     Agentic
首次提出     Fusion      兴起        技术        RAG
RAG概念      检索        成熟        发展        融合
```

## 1.2 RAG的核心组件

### 1.2.1 检索器（Retriever）

负责从知识库中检索与查询相关的文档：

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class DenseRetriever:
    """基于向量的稠密检索器"""

    def __init__(self, model_name="BAAI/bge-large-zh"):
        self.encoder = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):
        """编码文本为向量"""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True
        )
        return embeddings

    def search(self, query, documents, top_k=5):
        """检索最相关的文档"""
        query_vec = self.encode([query])
        doc_vecs = self.encode(documents)

        # 计算相似度
        scores = np.dot(query_vec, doc_vecs.T)[0]

        # 获取top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(idx, scores[idx]) for idx in top_indices]

# 使用示例
retriever = DenseRetriever()

documents = [
    "人工智能是计算机科学的一个分支",
    "深度学习是机器学习的一种方法",
    "Transformer是一种神经网络架构",
    "RAG结合检索和生成的优势"
]

results = retriever.search("什么是Transformer？", documents, top_k=2)
for idx, score in results:
    print(f"文档{idx}: {documents[idx]}, 相似度: {score:.4f}")
```

### 1.2.2 生成器（Generator）

基于检索结果生成回答的LLM：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGGenerator:
    """RAG生成器"""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, query, retrieved_contexts):
        """基于检索结果生成回答"""
        # 构建prompt
        prompt = f"""请根据以下信息回答问题。如果信息不足以回答，请说明。

检索到的相关信息：
{chr(10).join(retrieved_contexts)}

问题：{query}

回答："""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response
```

### 1.2.3 索引器（Indexer）

负责构建和管理知识库的索引：

```python
import faiss
import numpy as np
from typing import List, Tuple

class VectorIndexer:
    """向量索引器"""

    def __init__(self, dimension=1024, metric="cosine"):
        self.dimension = dimension
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # 内积（需要归一化）
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2距离

        self.documents = []
        self.id_to_doc = {}

    def add_documents(self, texts: List[str], embeddings: np.ndarray):
        """添加文档到索引"""
        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.documents.extend(texts)

        doc_id = len(self.documents) - len(texts)
        for i, text in enumerate(texts):
            self.id_to_doc[doc_id + i] = text

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """搜索最相似的文档"""
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        return list(zip(indices[0], scores[0]))

    def save(self, index_path: str, docs_path: str):
        """保存索引和文档"""
        faiss.write_index(self.index, index_path)

        import json
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents,
                "id_to_doc": self.id_to_doc
            }, f, ensure_ascii=False)

    def load(self, index_path: str, docs_path: str):
        """加载索引和文档"""
        self.index = faiss.read_index(index_path)

        import json
        with open(docs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.id_to_doc = data["id_to_doc"]
```

## 1.3 RAG的完整流程

```python
class SimpleRAG:
    """简单的RAG系统"""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.indexer = VectorIndexer(dimension=1024)

    def build_index(self, documents: List[str]):
        """构建知识库索引"""
        embeddings = self.retriever.encode(documents)
        self.indexer.add_documents(documents, embeddings)

    def query(self, user_query: str, top_k: int = 5) -> str:
        """处理用户查询"""
        # 1. 检索
        query_vec = self.retriever.encode([user_query])
        results = self.indexer.search(query_vec, top_k)

        # 2. 提取检索到的上下文
        retrieved_contexts = []
        for doc_idx, score in results:
            if doc_idx >= 0:  # 有效索引
                retrieved_contexts.append(self.indexer.documents[doc_idx])

        # 3. 生成回答
        response = self.generator.generate(user_query, retrieved_contexts)

        return response

    def add_documents(self, new_documents: List[str]):
        """增量添加文档"""
        embeddings = self.retriever.encode(new_documents)
        self.indexer.add_documents(new_documents, embeddings)


# 使用示例
rag = SimpleRAG(retriever, generator)

# 构建知识库
knowledge_base = [
    "Python是一种解释型、面向对象的高级编程语言。",
    "Python的设计哲学强调代码的可读性和简洁的语法。",
    "机器学习是人工智能的一个子领域，研究如何让计算机学习。",
    "深度学习是机器学习的一个分支，使用神经网络模拟人脑。",
]

rag.build_index(knowledge_base)

# 查询
response = rag.query("Python的特点是什么？")
print(response)
```

## 1.4 RAG的变体

### 1.4.1 基础RAG

最简单的RAG流程：检索 -> 生成

```python
# 参考上面的SimpleRAG实现
```

### 1.4.2 高级RAG

在基础RAG基础上增加了优化策略：

```
┌─────────────────────────────────────────────────────────────────┐
│                      高级RAG架构                                 │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   索引优化                                │   │
│   │  ├── 滑动窗口分块          ├── 元数据增强                │   │
│   │  ├── 层次化索引            └── MMR去重                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   查询优化                                │   │
│   │  ├── 查询重写          ├── 查询扩展                      │   │
│   │  ├── 子查询分解        └── 假设文档嵌入                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   检索优化                                │   │
│   │  ├── 混合检索            ├── 重排序                      │   │
│   │  ├── 向量检索+关键词     └── 迭代检索                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   生成优化                                │   │
│   │  ├── 上下文压缩        ├── 答案增强                      │   │
│   │  └── 多源融合          └── 反事实提示                    │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4.3 Agentic RAG

结合Agent能力的RAG系统：

```python
class AgenticRAG:
    """Agent驱动的RAG系统"""

    def __init__(self, llm, tools, retriever):
        self.llm = llm
        self.tools = tools  # 检索、搜索、计算等工具
        self.retriever = retriever
        self.history = []

    def query(self, user_query):
        """Agent处理查询"""
        self.history.append({"role": "user", "content": user_query})

        # Agent决定使用哪个工具
        plan = self._create_plan(user_query)

        # 执行检索
        context = self._execute_retrieval(plan)

        # 生成回答
        response = self._generate_response(user_query, context)

        self.history.append({"role": "assistant", "content": response})
        return response

    def _create_plan(self, query):
        """Agent规划检索策略"""
        prompt = f"""分析这个问题，确定最佳的检索策略：

问题：{query}

请确定：
1. 是否需要多次检索
2. 是否需要并行检索不同方面
3. 是否需要先进行查询扩展"""

        response = self.llm.generate(prompt)
        return json.loads(response)

    def _execute_retrieval(self, plan):
        """根据计划执行检索"""
        contexts = []
        for step in plan["steps"]:
            if step["type"] == "retrieve":
                results = self.retriever.search(step["query"])
                contexts.extend([r["content"] for r in results])
        return "\n".join(contexts)
```

## 1.5 RAG的评估

### 1.5.1 评估指标

| 指标类别 | 指标 | 说明 |
|---------|------|------|
| **检索质量** | Hit Rate, MRR, NDCG | 检索准确性和排序质量 |
| **生成质量** | ROUGE, BLEU | 与参考答案的相似度 |
| **端到端质量** | 回答准确性, 完整性 | 整体系统效果 |

### 1.5.2 检索评估实现

```python
import numpy as np

class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(self, query, retrieved_docs, relevant_docs, k_values=[1, 5, 10]):
        """评估检索质量"""
        metrics = {}

        for k in k_values:
            # Hit Rate @ k
            hits = sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs)
            metrics[f"Hit@{k}"] = hits / len(relevant_docs) if relevant_docs else 0

            # MRR @ k
            for i, doc in enumerate(retrieved_docs[:k]):
                if doc in relevant_docs:
                    metrics[f"MRR@{k}"] = 1 / (i + 1)
                    break
            else:
                metrics[f"MRR@{k}"] = 0

        return metrics

    def evaluate_generation(self, generated_answer, reference_answer):
        """评估生成质量"""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        scores = scorer.score(reference_answer, generated_answer)

        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }

    def run_full_evaluation(self, test_dataset):
        """运行完整评估"""
        all_retrieval_metrics = []
        all_generation_metrics = []

        for item in test_dataset:
            query = item["query"]
            retrieved = item["retrieved_docs"]
            relevant = item["relevant_docs"]
            generated = item["generated_answer"]
            reference = item["reference_answer"]

            retrieval_metrics = self.evaluate_retrieval(query, retrieved, relevant)
            generation_metrics = self.evaluate_generation(generated, reference)

            all_retrieval_metrics.append(retrieval_metrics)
            all_generation_metrics.append(generation_metrics)

        # 汇总结果
        avg_retrieval = {k: np.mean([m[k] for m in all_retrieval_metrics]) for k in all_retrieval_metrics[0]}
        avg_generation = {k: np.mean([m[k] for m in all_generation_metrics]) for k in all_generation_metrics[0]}

        return {
            "retrieval_metrics": avg_retrieval,
            "generation_metrics": avg_generation
        }
```

## 1.6 本章小结

本章介绍了RAG的基础知识：
- RAG的概念和优势
- 检索器、生成器、索引器三大组件
- 基础RAG和高级RAG的区别
- RAG系统的评估方法

## 练习题

1. 实现一个基于向量的简单检索器
2. 比较不同嵌入模型在检索任务上的效果
3. 构建一个简单的RAG系统并评估其性能
4. 了解RAG的优化策略和最新进展

## 参考资料

- [RAG原始论文](https://arxiv.org/abs/2005.11401) - Facebook RAG论文
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/) - LangChain文档
- [LlamaIndex RAG](https://docs.llamaindex.ai/) - LlamaIndex文档
- [BEIR数据集](https://github.com/beir-cellar/beir) - 检索评估基准
