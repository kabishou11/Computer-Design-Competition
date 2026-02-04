# 第一章：Agent基础概念

## 1.1 什么是Agent

Agent（智能体）是一种能够感知环境、做出决策并执行行动的自主系统。在大语言模型时代，LLM Agent指的是基于大语言模型构建的智能体，能够：

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Agent 核心能力                            │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
│   │  感知   │ →  │  推理   │ →  │  行动   │                     │
│   │ (输入)  │    │ (思考)  │    │ (执行)  │                     │
│   └─────────┘    └─────────┘    └─────────┘                     │
│        │              │              │                           │
│        ▼              ▼              ▼                           │
│   • 对话理解      • 任务规划      • API调用                       │
│   • 环境感知      • 逻辑推理      • 工具使用                       │
│   • 信息提取      • 知识检索      • 内容生成                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.1 Agent vs 传统程序

| 特性 | 传统程序 | Agent |
|-----|---------|-------|
| **工作流程** | 预设规则，固定流程 | 动态规划，灵活应对 |
| **输入处理** | 结构化输入 | 自然语言输入 |
| **输出** | 确定性结果 | 创造性输出 |
| **适应性** | 需人工修改规则 | 可自主学习 |
| **复杂度** | 规则复杂时难以维护 | 用自然语言描述需求 |

### 1.1.2 Agent的核心要素

```
Agent = 模型 + 工具 + 记忆 + 规划 + 推理
```

| 要素 | 说明 | 实现方式 |
|-----|------|---------|
| **模型** | 核心推理引擎 | GPT、Llama、Qwen等 |
| **工具** | 扩展能力边界 | 搜索、计算、数据库等 |
| **记忆** | 存储上下文信息 | 短期记忆、长期记忆 |
| **规划** | 分解任务为步骤 | CoT、ToT等 |
| **推理** | 逻辑思考过程 | ReAct、CoT等 |

## 1.2 Agent发展历程

```
时间线：
2020        2021        2022        2023        2024
  │           │           │           │           │
  │           │           │           │           │
  │           │           │           ▼           │
  │           │           │       GPT-4       Agent元年
  │           │           │       Plugins      多Agent
  │           │           │                     涌现
  │           │           │
  │           │           │
  ▼           ▼           ▼
早期LLM    提示工程    ChatGPT      LangChain
           兴起       诞生         Agent开发
```

## 1.3 Agent框架概览

### 1.3.1 主流框架对比

| 框架 | 开发方 | 特点 | 学习曲线 |
|-----|-------|------|---------|
| **LangChain** | LangChain AI | 功能全、生态丰富 | 中等 |
| **LangGraph** | LangChain AI | 状态管理、循环支持 | 较难 |
| **AgentScope** | 阿里巴巴 | 易用、中文友好 | 简单 |
| **AutoGen** | 微软 | 多Agent协作 | 中等 |
| **CrewAI** | CrewAI | 多Agent编排 | 简单 |
| **Dify** | -langchain | 低代码、可视化 | 简单 |

### 1.3.2 框架选择建议

```python
FRAMEWORK_GUIDE = {
    "learning": {
        "framework": "AgentScope",
        "reason": "简单易用，中文文档完善，适合入门"
    },
    "rapid_prototyping": {
        "framework": "Dify",
        "reason": "低代码平台，快速构建原型"
    },
    "production": {
        "framework": "LangGraph",
        "reason": "功能完善，支持复杂工作流"
    },
    "multi_agent": {
        "framework": "AutoGen/CrewAI",
        "reason": "多Agent协作能力强"
    }
}
```

## 1.4 简单Agent实现

### 1.4.1 基础Agent

```python
from typing import List, Dict, Callable
import json

class SimpleAgent:
    """简单Agent实现"""

    def __init__(self, llm_model, tools=None):
        self.llm = llm_model
        self.tools = tools or {}
        self.history = []

    def add_tool(self, name: str, func: Callable, description: str):
        """添加工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def run(self, user_input: str) -> str:
        """运行Agent"""
        # 记录用户输入
        self.history.append({"role": "user", "content": user_input})

        # 思考：是否需要使用工具
        response = self._think(user_input)

        # 记录响应
        self.history.append({"role": "assistant", "content": response})

        return response

    def _think(self, user_input: str) -> str:
        """思考过程"""
        # 构建提示
        prompt = self._build_prompt(user_input)

        # 调用LLM
        response = self.llm.generate(prompt)

        return response

    def _build_prompt(self, user_input: str) -> str:
        """构建提示"""
        tool_descriptions = []
        for name, info in self.tools.items():
            tool_descriptions.append(f"- {name}: {info['description']}")

        prompt = f"""你是一个智能助手，可以使用工具来帮助用户。

可用工具：
{chr(10).join(tool_descriptions)}

用户问题：{user_input}

请直接回答用户问题。如果需要使用工具，请清晰说明要使用的工具和参数。

回答："""

        return prompt


# 使用示例
class MockLLM:
    """模拟LLM"""
    def generate(self, prompt):
        return "我是智能助手，很高兴为您服务！"

agent = SimpleAgent(MockLLM())
print(agent.run("你好"))
```

### 1.4.2 带工具调用的Agent

```python
class ToolCallingAgent:
    """带工具调用的Agent"""

    def __init__(self, llm_model):
        self.llm = llm_model
        self.tools = {}
        self.history = []

    def add_tool(self, name: str, func: Callable, schema: Dict):
        """添加工具（带参数定义）"""
        self.tools[name] = {
            "function": func,
            "schema": schema
        }

    def run(self, user_input: str) -> str:
        """运行Agent"""
        self.history.append({"role": "user", "content": user_input})

        # 思考并决定是否调用工具
        response = self._reasoning(user_input)

        return response

    def _reasoning(self, user_input: str) -> str:
        """推理过程"""
        prompt = self._build_prompt(user_input)

        # 解析LLM响应中的工具调用
        llm_output = self.llm.generate(prompt)

        # 简单解析（实际应使用Function Calling格式）
        if "搜索" in user_input and "search" in self.tools:
            return self._handle_tool_call("search", {"query": user_input})

        return llm_output

    def _handle_tool_call(self, tool_name: str, args: Dict) -> str:
        """处理工具调用"""
        if tool_name not in self.tools:
            return f"错误：未知工具 {tool_name}"

        tool = self.tools[tool_name]
        result = tool["function"](**args)

        # 将工具结果返回给LLM生成最终回答
        final_response = self.llm.generate(
            f"用户问题：{self.history[-1]['content']}\n"
            f"工具结果：{result}\n"
            f"请根据工具结果回答用户。"
        )

        return final_response


# 定义工具
def search_web(query: str) -> str:
    """网络搜索工具"""
    # 实际实现可调用搜索API
    return f"搜索'{query}'的结果：..."

def calculate(expression: str) -> str:
    """计算工具"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except:
        return "计算错误"


# 使用示例
agent = ToolCallingAgent(MockLLM())
agent.add_tool("search", search_web, {
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索网络信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            }
        }
    }
})
agent.add_tool("calculate", calculate, {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            }
        }
    }
})

print(agent.run("计算 2+3*4"))
```

## 1.5 Agent的分类

### 1.5.1 按能力分类

| 类型 | 特点 | 示例 |
|-----|------|-----|
| **单Agent** | 独立完成任务 | 写作助手、问答机器人 |
| **多Agent** | 多Agent协作 | 辩论系统、团队协作 |
| **层次Agent** | 多层规划执行 | 任务分解、执行反馈 |

### 1.5.2 按应用场景分类

| 场景 | 应用 | 技术要点 |
|-----|------|---------|
| **对话** | 客服、闲聊 | 对话管理、情感识别 |
| **任务** | 办公助手、日程 | 任务规划、工具编排 |
| **研究** | 深度分析、报告 | 知识检索、多源整合 |
| **创作** | 写作、设计 | 创意生成、质量控制 |

### 1.5.3 按学习方式分类

| 类型 | 说明 | 训练方式 |
|-----|------|---------|
| **手工Agent** | 人工设计工作流 | 无需训练 |
| **微调Agent** | SFT/RLHF优化 | 需要标注数据 |
| **涌现Agent** | 通过提示涌现能力 | 零样本/少样本 |

## 1.6 本章小结

本章介绍了Agent的基础知识：
- Agent的定义和核心要素
- Agent与传统程序的区别
- Agent的发展历程
- 主流Agent框架对比
- 简单Agent的实现

## 练习题

1. 设计一个简单的问答Agent
2. 为Agent添加至少3个工具
3. 比较不同Agent框架的优缺点
4. 分析Agent在实际场景中的应用

## 参考资料

- [LangChain Agent Documentation](https://python.langchain.com/docs/modules/agents/)
- [AgentScope Documentation](https://agentscope.airesearch.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Agent推理范式
