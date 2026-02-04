# 高级RAG技术

## 1. 查询优化技术

### 1.1 查询改写

```python
class QueryRewriter:
    """查询改写器 - 使用LLM重写查询"""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """将口语化查询重写为更精确的形式"""
        prompt = f"""
请将以下用户查询重写为更精确、更适合信息检索的形式。

原始查询：{query}

要求：
1. 提取核心关键词
2. 扩展同义词
3. 明确查询意图
4. 保持原意不变

重写后的查询：
"""
        return self.llm.generate(prompt).strip()

    def decompose(self, query: str) -> List[str]:
        """将复杂查询分解为多个子查询"""
        prompt = f"""
将以下复杂查询分解为2-3个简单的独立查询。

原始查询：{query}

分解后的查询（每行一个）：
"""
        response = self.llm.generate(prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries

    def hyde(self, query: str) -> str:
        """HyDE (Hypothetical Document Embeddings)"""
        prompt = f"""
请假设你是该领域的专家，写一段关于"{query}"的详细回答。

假设文档：
"""
        hypothetical_doc = self.llm.generate(prompt)
        return hypothetical_doc
```

### 1.2 查询理解

```python
class QueryUnderstanding:
    """查询理解模块"""

    def __init__(self, llm):
        self.llm = llm

    def classify_intent(self, query: str) -> Dict:
        """分类查询意图"""
        prompt = f"""
分析以下查询的意图和特征：

查询：{query}

请分析：
1. 意图类型：事实查询/定义查询/列表查询/比较查询/解释查询
2. 复杂度：简单/中等/复杂
3. 时间范围：最新/历史/无时间限制
4. 专业程度：通俗/专业/学术

以JSON格式返回：
"""
        response = self.llm.generate(prompt)
        return json.loads(response)

    def extract_entities(self, query: str) -> List[Dict]:
        """提取查询中的实体"""
        prompt = f"""
从以下查询中提取关键实体（人物、地点、机构、概念等）：

查询：{query}

实体列表（JSON格式）：
"""
        response = self.llm.generate(prompt)
        return json.loads(response)
```

## 2. 高级检索策略

### 2.1 自适应检索

```python
class AdaptiveRetriever:
    """自适应检索器 - 根据查询类型选择检索策略"""

    def __init__(self, retrievers: Dict[str, BaseRetriever]):
        self.retrievers = retrievers

    def retrieve(self, query: str, query_type: str = None) -> List[Dict]:
        """根据查询类型选择检索器"""
        if query_type is None:
            query_type = self._classify_query(query)

        strategies = {
            "factual": ["vector", "bm25"],           # 事实查询：向量+BM25
            "comparative": ["vector", "kg"],          # 比较查询：向量+知识图谱
            "definitional": ["vector", "summary"],     # 定义查询：向量+摘要
            "list": ["bm25", "hybrid"],               # 列表查询：BM25+混合
            "analytical": ["hybrid", "rerank"],       # 分析查询：混合+重排序
        }

        selected = strategies.get(query_type, ["hybrid"])

        results = []
        for strategy in selected:
            if strategy in self.retrievers:
                results.extend(self.retrievers[strategy].retrieve(query))

        # 去重和排序
        return self._deduplicate_and_rerank(results)
```

### 2.2 迭代检索

```python
class IterativeRetriever:
    """迭代检索器 - 多轮检索增强"""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = 3

    def retrieve(self, query: str) -> List[str]:
        """迭代检索"""
        context = ""
        all_contexts = []

        for iteration in range(self.max_iterations):
            # 检索
            results = self.retriever.retrieve(query + context)

            # 检查是否需要继续
            if self._is_sufficient(results):
                break

            # 生成改进查询的提示
            prompt = f"""
基于以下检索结果，判断是否需要进一步检索：

检索结果：
{[r['text'] for r in results[:3]]}

如果需要进一步检索，请提供改进后的查询；如果已足够，请回复"DONE"。

改进后的查询或"DONE"：
"""
            improvement = self.generator.generate(prompt)

            if improvement.strip() == "DONE":
                break

            context = f"\n已检索：{improvement}"
            all_contexts.extend([r['text'] for r in results])

        return all_contexts

    def _is_sufficient(self, results: List[Dict]) -> bool:
        """检查检索结果是否足够"""
        if not results:
            return False
        # 检查最高相关性分数
        return results[0]['score'] > 0.9
```

### 2.3 图检索

```python
class GraphRetriever:
    """基于知识图谱的检索器"""

    def __init__(self, graph_db):
        self.graph = graph_db

    def retrieve(self, query: str, depth: int = 2) -> List[Dict]:
        """图遍历检索"""
        # 提取查询中的实体
        entities = self._extract_entities(query)

        results = []
        for entity in entities:
            # 从实体出发进行图遍历
            neighbors = self.graph.traverse(
                start_node=entity,
                depth=depth,
                relation_types=["related_to", "is_a", "part_of"]
            )
            results.extend(neighbors)

        return results

    def retrieve_with_reasoning(self, query: str) -> Tuple[List[Dict], List[str]]:
        """带推理路径的检索"""
        # 规划检索路径
        path = self._plan_retrieval_path(query)

        results = []
        reasoning = []

        for step in path:
            step_results = self._execute_step(step)
            results.extend(step_results)
            reasoning.append(f"步骤{len(reasoning)+1}: {step['description']}")

        return results, reasoning
```

## 3. RAG评估与优化

### 3.1 RAGAs评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

class RAGEvaluator:
    """RAG评估器"""

    def __init__(self, llm=None):
        self.llm = llm
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def evaluate(self, dataset: Dataset) -> Dict:
        """运行RAGAs评估"""
        from ragas import evaluate as ragas_evaluate

        results = ragas_evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm
        )

        return results

    def evaluate_with_custom_metrics(self, dataset: Dataset) -> Dict:
        """自定义指标评估"""
        from ragas.metrics import LLMEval

        custom_metrics = [
            LLMEval(
                name="answer_accuracy",
                definition="How accurate is the answer compared to ground truth?",
                evaluation_fn=self._evaluate_accuracy
            ),
            LLMEval(
                name="completeness",
                definition="How complete is the answer?",
                evaluation_fn=self._evaluate_completeness
            )
        ]

        return evaluate(dataset, metrics=custom_metrics)
```

### 3.2 检索质量分析

```python
class RetrievalAnalyzer:
    """检索质量分析器"""

    def __init__(self, retriever, gold_data: List[Dict]):
        self.retriever = retriever
        self.gold_data = gold_data

    def analyze_recall_at_k(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
        """计算Recall@K"""
        recalls = {}

        for k in k_values:
            hits = 0
            for item in self.gold_data:
                results = self.retriever.retrieve(item["query"], top_k=k)
                retrieved_ids = set(r["id"] for r in results)
                relevant_ids = set(item["relevant_ids"])

                if retrieved_ids & relevant_ids:
                    hits += 1

            recalls[k] = hits / len(self.gold_data)

        return recalls

    def analyze_mrr(self) -> float:
        """计算MRR (Mean Reciprocal Rank)"""
        reciprocal_ranks = []

        for item in self.gold_data:
            results = self.retriever.retrieve(item["query"])
            rank = None

            for i, r in enumerate(results):
                if r["id"] in item["relevant_ids"]:
                    rank = i + 1
                    break

            reciprocal_ranks.append(1 / rank if rank else 0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

## 4. RAG优化技巧

### 4.1 上下文压缩

```python
class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, llm, max_tokens=3000):
        self.llm = llm
        self.max_tokens = max_tokens

    def compress(self, contexts: List[str], query: str) -> List[str]:
        """压缩上下文"""
        # 计算当前token数
        current_tokens = sum(self.llm.count_tokens(c) for c in contexts)

        if current_tokens <= self.max_tokens:
            return contexts

        # 保留最相关的上下文
        relevance_scores = []
        for ctx in contexts:
            score = self._calculate_relevance(ctx, query)
            relevance_scores.append((ctx, score))

        # 按相关性排序
        relevance_scores.sort(key=lambda x: x[1], reverse=True)

        # 选择最重要的上下文
        compressed = []
        used_tokens = 0

        for ctx, score in relevance_scores:
            ctx_tokens = self.llm.count_tokens(ctx)
            if used_tokens + ctx_tokens <= self.max_tokens:
                compressed.append(ctx)
                used_tokens += ctx_tokens

        return compressed

    def compress_with_summary(self, contexts: List[str], query: str) -> str:
        """用摘要压缩长上下文"""
        summary_prompt = f"""
请根据以下查询，生成这些上下文的紧凑摘要：

查询：{query}

上下文：
{chr(10).join(contexts)}

摘要（保留关键信息，去除冗余）：
"""
        return self.llm.generate(summary_prompt)
```

### 4.2 提示优化

```python
class PromptOptimizer:
    """RAG提示优化器"""

    def __init__(self, llm):
        self.llm = llm

    def optimize_rag_prompt(
        self,
        task_description: str,
        context_template: str,
        output_format: str
    ) -> str:
        """优化RAG系统提示"""
        prompt = f"""
请为以下RAG系统设计最优提示：

任务描述：{task_description}

当前模板：
上下文：{{context}}
问题：{{question}}

期望输出格式：{output_format}

请提供：
1. 系统提示词
2. 上下文处理提示
3. 回答生成提示

请用中文回复：
"""
        return self.llm.generate(prompt)

    def generate_few_shot_examples(
        self,
        task: str,
        num_examples: int = 3
    ) -> List[Dict]:
        """生成少样本示例"""
        prompt = f"""
为以下任务生成{num_examples}个高质量的问答对：

任务：{task}

请提供JSON格式的示例列表：
"""
        response = self.llm.generate(prompt)
        return json.loads(response)
```

## 5. 最新RAG技术趋势

### 5.1 Agentic RAG

```python
class AgenticRAG:
    """Agent驱动的RAG系统"""

    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = []

    def query(self, user_query: str) -> Dict:
        """Agent处理复杂查询"""
        # Agent规划检索策略
        plan = self._plan(user_query)

        # 执行检索
        context = self._execute_plan(plan)

        # 生成回答
        answer = self._generate_answer(user_query, context)

        return {
            "answer": answer,
            "context": context,
            "plan": plan,
            "steps": self.memory
        }

    def _plan(self, query: str) -> List[Dict]:
        """Agent规划检索策略"""
        prompt = f"""
分析以下查询，制定检索策略：

查询：{query}

可用工具：{list(self.tools.keys())}

请JSON格式返回检索步骤：
"""
        response = self.llm.generate(prompt)
        return json.loads(response)

    def _execute_plan(self, plan: List[Dict]) -> str:
        """执行检索计划"""
        contexts = []

        for step in plan:
            tool_name = step["tool"]
            if tool_name in self.tools:
                result = self.tools[tool_name].execute(step["params"])
                contexts.append(result)
                self.memory.append({
                    "step": step,
                    "result": result
                })

        return "\n".join(contexts)
```

### 5.2 Self-RAG

```python
class SelfRAG:
    """Self-RAG: 自反思RAG"""

    def __init__(self, rag_model, critic_model):
        self.rag_model = rag_model  # 生成模型
        self.critic_model = critic_model  # 批判模型

    def generate_with_reflection(
        self,
        query: str,
        retrieved_contexts: List[str],
        max_tokens: int = 512
    ) -> str:
        """自反思生成"""
        # 1. 判断是否需要检索
        if not self._is_retrieval_needed(query):
            return self.rag_model.generate(query, max_tokens)

        # 2. 生成候选答案
        candidates = self._generate_candidates(query, retrieved_contexts)

        # 3. 批判每个候选
        critiques = []
        for candidate in candidates:
            critique = self._critique(query, candidate)
            critiques.append(critique)

        # 4. 选择最佳答案
        best = self._select_best(candidates, critiques)

        return best

    def _is_retrieval_needed(self, query: str) -> bool:
        """判断是否需要检索"""
        prompt = f"""
判断以下查询是否需要额外检索才能回答：

查询：{query}

如果需要检索请回复"YES"，否则回复"NO"：
"""
        response = self.critic_model.generate(prompt)
        return "YES" in response.upper()

    def _critique(self, query: str, answer: str) -> Dict:
        """批判答案"""
        prompt = f"""
请批判以下答案对查询的质量：

查询：{query}

答案：{answer}

请JSON格式返回：
{{
    "relevance": 1-5分,
    "faithfulness": 1-5分,
    "utility": 1-5分,
    "critique": "详细批评意见",
    "improvement": "改进建议"
}}
"""
        response = self.critic_model.generate(prompt)
        return json.loads(response)
```

## 6. 本章小结

本章介绍了高级RAG技术：
- 查询优化（改写、理解、HyDE）
- 自适应检索策略
- 迭代检索和图检索
- RAGAs评估方法
- Agentic RAG和Self-RAG

## 参考资料

- [RAG Survey](https://arxiv.org/abs/2312.10997) - RAG综述
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511) - Self-RAG
- [HyDE Paper](https://arxiv.org/abs/2212.10496) - HyDE
- [RAGAs](https://github.com/explodinggradients/ragas) - RAG评估库
