# 第二章：ReAct推理范式

## 2.1 ReAct概述

### 2.1.1 什么是ReAct

ReAct（Reasoning + Acting）是一种结合推理和行动的Agent范式，由Google Research提出。其核心思想是：

```
思考（Reasoning）→ 行动（Acting）→ 观察（Observation）→ 思考（Reasoning）→ ...
```

### 2.1.2 ReAct vs 其他范式

| 范式 | 特点 | 优势 | 劣势 |
|-----|------|------|------|
| **CoT** | 纯推理 | 逻辑清晰 | 无法行动 |
| **ReAct** | 推理+行动 | 交互能力强 | 可能过度行动 |
| **ToT** | 树搜索推理 | 全局最优 | 计算开销大 |
| **PoT** | 程序化推理 | 可验证 | 依赖程序生成 |

### 2.1.3 ReAct工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReAct 循环                                  │
│                                                                 │
│   ┌──────────────────────────────────────────────────────┐     │
│   │   Thought: 思考当前状态和下一步行动                    │     │
│   │   Action:   执行具体行动                              │     │
│   │   Observation: 观察行动结果                            │     │
│   └──────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│                   ┌───────────────┐                             │
│                   │   达到终止条件？ │                          │
│                   │ (任务完成/达到上限)│                        │
│                   └───────┬───────┘                             │
│                           │                                     │
│               ┌───────────┴───────────┐                          │
│               │ 是                   │ 否                       │
│               ▼                       ▼                          │
│       ┌─────────────┐        ┌──────────────────┐              │
│       │   返回结果   │        │  继续ReAct循环    │              │
│       └─────────────┘        └──────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 ReAct Prompt设计

### 2.2.1 Prompt模板

```python
REACT_PROMPT = """
你是一个智能助手，通过思考和行动来解决问题。

请按照以下格式思考和行动：

Thought 1: [描述当前的想法和要采取的行动]
Action 1: [执行的行动，如：search[查询内容]或calculate[表达式]]
Observation 1: [行动的结果]

Thought 2: [基于观察结果的思考]
Action 2: [下一个行动]
Observation 2: [结果]
...

Thought N: [最终推理]
Answer: [最终答案]

可用工具：
- search: 搜索网络信息
- calculate: 执行数学计算
- lookup: 在知识库中查找信息

现在开始解决问题：
用户问题：{user_question}
"""
```

### 2.2.2 ReAct Agent实现

```python
from typing import List, Dict, Tuple
import json

class ReActAgent:
    """基于ReAct范式的Agent"""

    def __init__(self, llm_model, max_iterations=10):
        self.llm = llm_model
        self.max_iterations = max_iterations
        self.tools = {}

    def add_tool(self, name: str, func: Callable, description: str):
        """添加工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def run(self, question: str) -> Tuple[str, List[Dict]]:
        """运行Agent"""
        history = []
        context = ""

        for iteration in range(self.max_iterations):
            # 构建思考提示
            prompt = self._build_thought_prompt(question, context, history)

            # 获取思考和行动
            llm_output = self.llm.generate(prompt)

            # 解析输出
            thought, action, action_input = self._parse_output(llm_output)

            # 记录思考
            step_info = {
                "step": iteration + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input
            }
            history.append(step_info)

            # 执行行动
            if action == "FINISH":
                # 任务完成
                return action_input, history

            if action and action in self.tools:
                # 执行工具调用
                result = self.tools[action]["function"](action_input)
                context += f"\n观察: {result}"
                step_info["observation"] = result
            else:
                # 未知行动，尝试直接回答
                context += f"\n观察: 无法执行行动 '{action}'"
                action = "FINISH"
                action_input = llm_output.split("Answer:")[-1].strip() if "Answer:" in llm_output else llm_output

        # 达到最大迭代次数
        final_answer = self._finalize_answer(question, history)
        return final_answer, history

    def _build_thought_prompt(self, question: str, context: str, history: List[Dict]) -> str:
        """构建思考提示"""
        tool_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        history_text = ""
        for step in history:
            history_text += f"\nThought {step['step']}: {step['thought']}"
            history_text += f"\nAction {step['step']}: {step['action']}[{step['action_input']}]"
            history_text += f"\nObservation {step['step']}: {step.get('observation', '')}"

        prompt = f"""
你是一个智能助手，通过思考和行动来解决问题。

可用工具：
{tool_desc}

历史步骤：
{history_text}

当前上下文：{context}

请按照以下格式继续思考和行动：
Thought: [你的推理]
Action: [工具名称，或FINISH]
Action Input: [工具参数，如果适用]
Answer: [最终答案（仅当Action为FINISH时填写）]

用户问题：{question}

请开始思考：
"""
        return prompt

    def _parse_output(self, output: str) -> Tuple[str, str, str]:
        """解析LLM输出"""
        lines = output.strip().split('\n')

        thought = ""
        action = ""
        action_input = ""
        answer = ""

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()

        if action == "FINISH":
            return thought, "FINISH", answer

        return thought, action, action_input

    def _finalize_answer(self, question: str, history: List[Dict]) -> str:
        """生成最终答案"""
        prompt = f"""
基于以下推理历史，为用户问题生成最终答案：

用户问题：{question}

推理历史：
{chr(10).join([f"步骤{i+1}: {step['thought']}" for i, step in enumerate(history)])}

请简洁回答用户问题：
"""
        return self.llm.generate(prompt)
```

### 2.2.3 使用示例

```python
# 定义工具
def search_knowledge_base(query: str) -> str:
    """知识库搜索"""
    knowledge = {
        "人工智能": "人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
        "机器学习": "机器学习是人工智能的一个子领域，研究如何让计算机通过数据学习并改进。",
        "深度学习": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。"
    }
    return knowledge.get(query, "未找到相关信息")

def calculator(expression: str) -> str:
    """计算器"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"


# 创建Agent
agent = ReActAgent(MockLLM(), max_iterations=5)
agent.add_tool("search", search_knowledge_base, "搜索知识库")
agent.add_tool("calculate", calculator, "执行数学计算")

# 运行
question = "深度学习和机器学习有什么关系？"
answer, history = agent.run(question)

print("=" * 50)
print(f"问题：{question}")
print("=" * 50)
for step in history:
    print(f"步骤 {step['step']}:")
    print(f"  思考: {step['thought']}")
    print(f"  行动: {step['action']}[{step['action_input']}]")
    print(f"  观察: {step.get('observation', '')}")
print("=" * 50)
print(f"最终答案：{answer}")
```

## 2.3 ReAct变体

### 2.3.1 ReAct+Self-Consistency

```python
class ReActWithConsistency(ReActAgent):
    """带一致性检验的ReAct"""

    def __init__(self, llm_model, num_samples=3, max_iterations=5):
        super().__init__(llm_model, max_iterations)
        self.num_samples = num_samples

    def run(self, question: str) -> str:
        """多次采样并投票"""
        answers = []

        for _ in range(self.num_samples):
            answer, _ = super().run(question)
            answers.append(answer)

        # 投票选择最常见的答案
        return self._vote(answers)

    def _vote(self, answers: List[str]) -> str:
        """多数投票"""
        from collections import Counter
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
```

### 2.3.2 Plan-and-Execute

```python
class PlanAndExecuteAgent:
    """计划-执行模式Agent"""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.tools = {}

    def add_tool(self, name: str, func: Callable):
        """添加工具"""
        self.tools[name] = func

    def run(self, question: str) -> str:
        """运行Agent"""
        # 1. 制定计划
        plan = self._create_plan(question)

        # 2. 执行计划
        results = []
        for step in plan:
            result = self._execute_step(step)
            results.append(result)

        # 3. 生成最终答案
        answer = self._synthesize(question, plan, results)

        return answer

    def _create_plan(self, question: str) -> List[str]:
        """创建执行计划"""
        prompt = f"""
将以下问题分解为多个执行步骤：

问题：{question}

步骤（用序号标记）：
1. [第一个步骤]
2. [第二个步骤]
...
"""
        response = self.llm.generate(prompt)

        # 解析步骤
        steps = []
        for line in response.split('\n'):
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                step = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                step = step.lstrip('-. ')
                steps.append(step)

        return steps

    def _execute_step(self, step: str) -> str:
        """执行单个步骤"""
        prompt = f"""
执行以下步骤，如果需要使用工具，请明确说明：

步骤：{step}

可用工具：{', '.join(self.tools.keys())}

执行结果：
"""
        response = self.llm.generate(prompt)

        # 解析工具调用
        if '[' in response and ']' in response:
            tool_name = response.split('[')[1].split(']')[0]
            tool_input = response.split('[')[1].split(']')[0].split('[')[-1].split(']')[0] if '[' in response else ""

            if tool_name in self.tools:
                return self.tools[tool_name](tool_input)

        return response

    def _synthesize(self, question: str, plan: List[str], results: List[str]) -> str:
        """综合结果生成答案"""
        prompt = f"""
基于以下执行结果，回答用户问题：

问题：{question}

执行步骤和结果：
{chr(10).join([f"{i+1}. {s}: {r}" for i, (s, r) in enumerate(zip(plan, results))])}

最终答案：
"""
        return self.llm.generate(prompt)
```

## 2.4 ReAct评估

### 2.4.1 评估指标

| 指标 | 说明 | 计算方式 |
|-----|------|---------|
| **成功率** | 任务完成比例 | 完成数/总数 |
| **步骤数** | 平均执行步骤 | 总步数/任务数 |
| **回溯率** | 需要回溯的比例 | 回溯数/总数 |
| **答案质量** | 最终答案质量 | 人工/自动评估 |

### 2.4.2 评估实现

```python
class ReActEvaluator:
    """ReAct评估器"""

    def __init__(self, agent, eval_dataset):
        self.agent = agent
        self.dataset = eval_dataset

    def evaluate(self):
        """运行评估"""
        results = []

        for item in self.dataset:
            question = item["question"]
            expected = item["expected_answer"]

            answer, history = self.agent.run(question)

            # 计算评估指标
            metrics = self._compute_metrics(answer, expected, history)

            results.append({
                "question": question,
                "answer": answer,
                "expected": expected,
                "history": history,
                "metrics": metrics
            })

        return self._summarize(results)

    def _compute_metrics(self, answer: str, expected: str, history: List[Dict]) -> Dict:
        """计算各项指标"""
        # 成功率（简化：检查关键信息是否包含）
        success = self._check_success(answer, expected)

        # 步骤数
        num_steps = len(history)

        # 是否使用工具
        uses_tools = any(
            step.get("action") in ["search", "calculate", "lookup"]
            for step in history
        )

        return {
            "success": success,
            "num_steps": num_steps,
            "uses_tools": uses_tools
        }

    def _check_success(self, answer: str, expected: str) -> bool:
        """检查是否成功回答"""
        # 简化：检查关键词
        expected_keywords = set(expected.lower().split())
        answer_lower = answer.lower()
        matches = sum(1 for kw in expected_keywords if kw in answer_lower)
        return matches / len(expected_keywords) > 0.5 if expected_keywords else False

    def _summarize(self, results: List[Dict]) -> Dict:
        """汇总评估结果"""
        total = len(results)
        successes = sum(1 for r in results if r["metrics"]["success"])
        avg_steps = np.mean([r["metrics"]["num_steps"] for r in results])

        return {
            "total_tasks": total,
            "success_rate": successes / total,
            "avg_steps": avg_steps,
            "success_examples": [r for r in results if r["metrics"]["success"]][:5],
            "failure_examples": [r for r in results if not r["metrics"]["success"]][:5]
        }
```

## 2.5 本章小结

本章介绍了ReAct推理范式：
- ReAct的核心思想（思考→行动→观察→思考）
- ReAct Prompt设计
- ReAct Agent实现
- ReAct变体（Plan-and-Execute等）
- ReAct评估方法

## 练习题

1. 实现一个完整的ReAct Agent
2. 为Agent添加多种工具并测试
3. 比较CoT和ReAct在不同任务上的表现
4. 设计一个评估ReAct性能的测试集

## 参考资料

- [ReAct论文](https://arxiv.org/abs/2210.03629) - 原始论文
- [LangChain ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react) - LangChain实现
- [Agent Reasoning Patterns](https://www.promptingguide.ai/techniques/react) - 推理模式指南
