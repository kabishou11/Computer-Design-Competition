# 高级Agent技术

## 1. 记忆系统

### 1.1 记忆类型

```
Agent记忆架构

┌─────────────────────────────────────────────────────────┐
│                      Agent Memory                       │
├─────────────────┬─────────────────────────────────────┤
│   短期记忆      │   长期记忆                          │
│  (Working)      │   (Long-term)                       │
├─────────────────┼─────────────────────────────────────┤
│ • 当前对话      │ • 用户偏好                          │
│ • 即时上下文    │ • 历史交互                          │
│ • 临时计算结果  │ • 学习到的知识                      │
├─────────────────┼─────────────────────────────────────┤
│ 容量: 有限      │ 容量: 无限                          │
│ 生命周期: 会话  │ 生命周期: 持久                     │
│ 实现: Attention │ 实现: VectorDB + Summary           │
└─────────────────┴─────────────────────────────────────┘
```

### 1.2 记忆实现

```python
from datetime import datetime
from typing import List, Dict
import json

class MemoryManager:
    """记忆管理器"""

    def __init__(self, vector_store=None):
        self.short_term = []  # 短期记忆
        self.vector_store = vector_store  # 长期记忆向量库

    def add_to_short_term(self, content: str, role: str = "user"):
        """添加短期记忆"""
        self.short_term.append({
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat()
        })

        # 限制短期记忆大小
        if len(self.short_term) > 10:
            self._consolidate_memory()

    def _consolidate_memory(self):
        """记忆整合"""
        if not self.vector_store:
            return

        # 摘要化最早的记忆
        old_memories = self.short_term[:-5]
        summary = self._summarize(old_memories)

        # 存入长期记忆
        self.vector_store.add(summary, metadata={"type": "consolidated_memory"})

        # 保留最近5轮
        self.short_term = self.short_term[-5:]

    def _summarize(self, memories: List[Dict]) -> str:
        """摘要记忆"""
        text = "\n".join([m["content"] for m in memories])
        # 调用LLM进行摘要
        return f"[历史摘要]{text[:200]}..."

    def retrieve_relevant_memory(self, query: str) -> List[str]:
        """检索相关记忆"""
        if self.vector_store:
            results = self.vector_store.search(query)
            return [r["text"] for r in results]
        return []

    def get_full_context(self) -> str:
        """获取完整上下文"""
        # 长期记忆
        relevant = self.retrieve_relevant_memory("")

        # 短期记忆
        recent = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in self.short_term
        ])

        return f"相关历史：{relevant}\n\n最近对话：{recent}"
```

### 1.3 记忆增强Agent

```python
class MemoryAugmentedAgent:
    """记忆增强Agent"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.memory = MemoryManager(vector_store)

    def run(self, user_input: str) -> str:
        """运行Agent"""
        # 检索相关记忆
        relevant_memory = self.memory.retrieve_relevant_memory(user_input)

        # 构建上下文
        context = f"相关记忆：{relevant_memory}\n\n当前对话：{user_input}"

        # 生成回答
        response = self.llm.generate(context)

        # 保存对话
        self.memory.add_to_short_term(user_input, "user")
        self.memory.add_to_short_term(response, "assistant")

        return response
```

## 2. 多Agent协作

### 2.1 多Agent架构

```
多Agent协作模式

模式1: 层级式
    ┌────────────┐
    │   Manager  │  ← 协调分配任务
    └─────┬──────┘
          │
    ┌─────┴──────┐
    │   │   │     │
    ▼   ▼   ▼     │
  Worker1 Worker2  │  ← 执行具体任务
                 │

模式2: 协作式
    ┌─────┬─────┬─────┐
    │     │     │     │
    Agent Agent Agent  ← 平等协作、讨论
    │     │     │     │
    └─────┴─────┴─────┘

模式3: 辩论式
    ┌──────────────┐
    │  Agent A ↔ Agent B  ← 辩论
    └──────┬───────┘
           │
           ▼
      综合决策
```

### 2.2 Agent协作实现

```python
from typing import List
from dataclasses import dataclass

@dataclass
class AgentMessage:
    """Agent消息"""
    from_agent: str
    to_agent: str
    content: str
    timestamp: float

class MultiAgentSystem:
    """多Agent系统"""

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.message_board = []
        self.shared_context = {}

    def broadcast(self, from_agent: str, message: str):
        """广播消息"""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent="all",
            content=message,
            timestamp=datetime.now().timestamp()
        )
        self.message_board.append(msg)

    def send_message(self, from_agent: str, to_agent: str, message: str):
        """发送消息"""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=message,
            timestamp=datetime.now().timestamp()
        )
        self.message_board.append(msg)
        self.agents[to_agent].receive(msg)

    def collaborate_on_task(self, task: str) -> str:
        """协作完成复杂任务"""
        # 1. 任务分解
        subtasks = self._decompose_task(task)

        # 2. 分配任务
        assignments = self._assign_subtasks(subtasks)

        # 3. 并行执行
        results = self._execute_in_parallel(assignments)

        # 4. 整合结果
        final_result = self._integrate_results(results)

        return final_result

    def _decompose_task(self, task: str) -> List[Dict]:
        """分解任务"""
        prompt = f"""
将以下复杂任务分解为多个子任务：

任务：{task}

子任务列表（JSON格式）：
"""
        return json.loads(self.llm.generate(prompt))

    def _assign_subtasks(self, subtasks: List[Dict]) -> Dict[str, List]:
        """分配子任务"""
        # 根据Agent能力分配
        assignments = {}
        for subtask in subtasks:
            best_agent = self._select_best_agent(subtask)
            if best_agent not in assignments:
                assignments[best_agent] = []
            assignments[best_agent].append(subtask)
        return assignments

    def _execute_in_parallel(self, assignments: Dict[str, List]) -> Dict:
        """并行执行"""
        import concurrent.futures

        results = {}

        def execute_for_agent(agent_name, subtasks):
            return agent_name, [
                self.agents[agent_name].run(subtask["description"])
                for subtask in subtasks
            ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(execute_for_agent, agent, tasks)
                for agent, tasks in assignments.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                agent_name, agent_results = future.result()
                results[agent_name] = agent_results

        return results
```

### 2.3 辩论系统

```python
class DebateSystem:
    """辩论系统"""

    def __init__(self, agent_a: BaseAgent, agent_b: BaseAgent):
        self.proponent = agent_a  # 正方
        self.opponent = agent_b   # 反方
        self.debate_history = []

    def debate(self, topic: str, rounds: int = 3) -> Dict:
        """进行辩论"""
        result = {
            "topic": topic,
            "rounds": [],
            "final_position": None
        }

        # 正方开场
        opening = self.proponent.run(f"请就以下观点发表开场陈述：{topic}")
        self.debate_history.append({"speaker": "proponent", "content": opening})

        # 多轮辩论
        for round_idx in range(rounds):
            # 反方反驳
            rebuttal = self.opponent.run(
                f"请反驳以下观点：{opening}\n\n辩论历史：{self.debate_history}"
            )
            self.debate_history.append({"speaker": "opponent", "content": rebuttal})

            # 正方回应
            response = self.proponent.run(
                f"请回应以下反驳：{rebuttal}\n\n辩论历史：{self.debate_history}"
            )
            self.debate_history.append({"speaker": "proponent", "content": response})

            result["rounds"].append({
                "round": round_idx + 1,
                "rebuttal": rebuttal,
                "response": response
            })

        # 综合双方观点
        final_position = self._synthesize(topic)
        result["final_position"] = final_position

        return result

    def _synthesize(self, topic: str) -> str:
        """综合观点"""
        debate_text = "\n".join([
            f"{d['speaker']}: {d['content']}"
            for d in self.debate_history
        ])

        prompt = f"""
基于以下辩论内容，给出一个综合观点：

辩论内容：{debate_text}

原始话题：{topic}

综合观点（考虑双方论点）：
"""
        return self.llm.generate(prompt)
```

## 3. 工具构建

### 3.1 自定义工具

```python
from typing import Dict, Any
from langchain.tools import BaseTool

class CustomSearchTool(BaseTool):
    """自定义搜索工具"""
    name = "custom_search"
    description = "搜索指定来源的信息"

    def __init__(self, search_engine):
        self.search_engine = search_engine

    def _run(self, query: str, source: str = "general") -> str:
        """执行搜索"""
        results = self.search_engine.search(query, source=source)
        return json.dumps(results[:5], ensure_ascii=False)

    async def _arun(self, query: str, source: str = "general") -> str:
        """异步执行"""
        return self._run(query, source)


class DataAnalysisTool(BaseTool):
    """数据分析工具"""
    name = "data_analysis"
    description = "对数据进行分析和统计"

    def __init__(self, data_processor):
        self.processor = data_processor

    def _run(self, data: str, analysis_type: str = "summary") -> str:
        """执行分析"""
        import pandas as pd

        df = pd.read_json(data)
        if analysis_type == "summary":
            return df.describe().to_json()
        elif analysis_type == "correlation":
            return df.corr().to_json()
        else:
            return df.head().to_json()
```

### 3.2 工具注册

```python
class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self.tools = {}

    def register(self, name: str, tool: BaseTool):
        """注册工具"""
        self.tools[name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [
            {"name": name, "description": tool.description}
            for name, tool in self.tools.items()
        ]

    def create_tool_from_function(self, func: Callable, name: str = None):
        """从函数创建工具"""
        tool_name = name or func.__name__

        class FunctionTool(BaseTool):
            name = tool_name
            description = func.__doc__ or f"执行{tool_name}函数"

            def _run(self, *args, **kwargs):
                return func(*args, **kwargs)

            async def _arun(self, *args, **kwargs):
                return func(*args, **kwargs)

        self.register(tool_name, FunctionTool())
```

## 4. Agent评估

### 4.1 评估维度

| 维度 | 指标 | 说明 |
|-----|------|------|
| **任务完成度** | 成功率 | 正确完成任务的比例 |
| **规划能力** | 步数效率 | 完成任务的步数 |
| **工具使用** | 调用准确率 | 工具调用正确率 |
| **对话质量** | 流畅度、相关性 | 对话的自然程度 |
| **鲁棒性** | 错误恢复率 | 遇到错误后的恢复能力 |

### 4.2 评估实现

```python
class AgentEvaluator:
    """Agent评估器"""

    def __init__(self, agent, eval_dataset):
        self.agent = agent
        self.dataset = eval_dataset

    def evaluate(self) -> Dict:
        """运行评估"""
        results = []

        for item in self.dataset:
            # 运行Agent
            response, history = self.agent.run(item["query"])

            # 计算各项指标
            metrics = {
                "task_success": self._check_success(response, item["expected"]),
                "step_efficiency": len(history),
                "tool_usage": self._evaluate_tool_usage(history),
                "response_quality": self._evaluate_quality(response, item["query"])
            }

            results.append({
                "query": item["query"],
                "response": response,
                "metrics": metrics
            })

        return self._aggregate_results(results)

    def _check_success(self, response: str, expected: str) -> bool:
        """检查任务是否成功"""
        # 使用LLM判断
        prompt = f"""
判断以下回答是否正确回答了用户问题：

问题：{expected}

回答：{response}

请仅回复"YES"或"NO"：
"""
        result = self.agent.llm.generate(prompt)
        return "YES" in result.upper()

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """汇总结果"""
        total = len(results)
        success_count = sum(1 for r in results if r["metrics"]["task_success"])
        avg_steps = sum(len(r["response"].split()) for r in results) / total

        return {
            "success_rate": success_count / total,
            "avg_response_length": avg_steps,
            "detailed_results": results
        }
```

## 5. 最新发展趋势

### 5.1 2024-2025 Agent趋势

| 趋势 | 说明 | 代表工作 |
|-----|------|---------|
| **Agentic Workflow** | Agent作为工作流编排 | OpenAI GPTs |
| **Multi-Modal Agent** | 处理多模态输入 | GPT-4o |
| **Self-Improving Agent** | 从错误中学习 | Reflexion |
| **Collaborative Agents** | 多Agent协作 | AutoGen, CrewAI |
| **Edge Agent** | 边缘设备部署 | 本地LLM + Agent |

### 5.2 竞赛项目建议

对于计算机设计大赛，Agent方向推荐项目：

| 项目类型 | 技术要点 | 推荐度 |
|---------|---------|-------|
| 自动化工作流 | Agent + 多工具编排 | 高 |
| 研究助手 | Agent + 搜索 + RAG | 高 |
| 协作机器人 | 多Agent辩论/讨论 | 中 |
| 领域专家Agent | Agent + 专业知识 | 高 |
| 创意生成系统 | Agent + 创意模板 | 中 |

## 6. 本章小结

本章介绍了高级Agent技术：
- 记忆系统（短期+长期）
- 多Agent协作（层级、协作、辩论）
- 自定义工具构建
- Agent评估方法
- 最新发展趋势

## 参考资料

- [LangChain Agent](https://python.langchain.com/docs/modules/agents/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://www.crewai.com/)
- [Agent Evaluation](https://github.com/Claude-Agent/agent-evals)
