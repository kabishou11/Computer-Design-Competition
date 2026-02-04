# 第三章：Agent框架实践

## 3.1 LangChain Agent

### 3.1.1 工具定义

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 定义工具
@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库获取信息"""
    knowledge = {
        "人工智能": "AI是计算机科学分支，研究创建智能机器",
        "机器学习": "ML是AI子领域，让计算机从数据学习",
        "深度学习": "DL是ML分支，使用多层神经网络"
    }
    return knowledge.get(query, f"未找到'{query}'的相关信息")

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### 3.1.2 创建Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建Agent
agent = initialize_agent(
    tools=[search_knowledge_base, calculator, get_current_time],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=10
)

# 运行Agent
response = agent.run("人工智能的定义是什么？计算一下 25*4+10 的结果")
print(response)
```

### 3.1.3 自定义Agent

```python
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Optional, Type

class CustomSearchTool(BaseTool):
    """自定义搜索工具"""
    name = "custom_search"
    description = "搜索网络信息，返回相关结果"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # 实现搜索逻辑
        return f"搜索'{query}'的结果"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class CustomAgent:
    """自定义Agent"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = []

    def run(self, user_input: str) -> str:
        """运行Agent"""
        self.memory.append({"role": "user", "content": user_input})

        # 构建提示
        prompt = self._build_prompt(user_input)

        # 调用LLM
        response = self.llm.invoke(prompt)

        # 解析并执行工具调用
        result = self._handle_response(response)

        self.memory.append({"role": "assistant", "content": result})

        return result

    def _build_prompt(self, user_input: str) -> str:
        """构建提示"""
        tool_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""
你是一个智能助手，可以使用工具来回答问题。

可用工具：
{tool_desc}

对话历史：
{chr(10).join([f"{m['role']}: {m['content']}" for m in self.memory[-5:]])}

用户问题：{user_input}

请回答用户问题。如果需要使用工具，请明确使用[tool_name:参数]格式。
"""

    def _handle_response(self, response) -> str:
        """处理LLM响应"""
        content = response.content if hasattr(response, 'content') else str(response)

        # 检查工具调用
        if '[' in content and ']' in content:
            import re
            tool_calls = re.findall(r'\[(\w+):([^\]]+)\]', content)

            for tool_name, tool_input in tool_calls:
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name]._run(tool_input)
                    content = content.replace(
                        f"[{tool_name}:{tool_input}]",
                        f"\n工具'{tool_name}'的结果：{tool_result}\n"
                    )

        return content
```

## 3.2 LangGraph Agent

### 3.2.1 状态图构建

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

# 定义状态
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_task: str
    tool_calls: List[Dict]
    results: List[str]
    final_answer: str

# 定义节点
def llm_node(state: AgentState) -> AgentState:
    """LLM推理节点"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 构建消息
    messages = state["messages"]
    response = llm.invoke(messages)

    state["messages"].append(response)
    return state

def tool_node(state: AgentState) -> AgentState:
    """工具执行节点"""
    import json

    # 解析工具调用
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, 'tool_calls', [])

    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        # 执行工具（简化）
        results.append(f"执行{tool_name}的结果")

    state["tool_calls"] = tool_calls
    state["results"] = results
    return state

def should_continue(state: AgentState) -> str:
    """决定是否继续"""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "execute_tools"
    return "end"


# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)

# 添加边
workflow.set_entry_point("llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "execute_tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "llm")

# 编译图
app = workflow.compile()
```

### 3.2.2 带记忆的Agent

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

class MemoryAgent:
    """带记忆的Agent"""

    def __init__(self, llm):
        self.llm = llm
        self.memory = MemorySaver()

    def create_graph(self):
        """创建带记忆的图"""
        def chat_node(state):
            messages = state["messages"]
            response = self.llm.invoke(messages)
            return {"messages": messages + [response]}

        workflow = StateGraph(dict)
        workflow.add_node("chat", chat_node)
        workflow.set_entry_point("chat")
        workflow.add_conditional_edges("chat", lambda x: END)

        return workflow.compile(checkpointer=self.memory)

    def run(self, user_input: str, thread_id: str = "default"):
        """运行Agent"""
        app = self.create_graph()

        config = {"configurable": {"thread_id": thread_id}}

        state = {"messages": [HumanMessage(content=user_input)]}
        result = app.invoke(state, config=config)

        return result["messages"][-1].content
```

## 3.3 AgentScope实践

### 3.3.1 简单Agent

```python
# pip install agentscope

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.service import ServiceToolkits

class SimpleAgent(AgentBase):
    """简单Agent"""

    def __init__(self, name: str):
        super().__init__(name=name, host="localhost", port=8080)

    def reply(self, x: Msg) -> Msg:
        """回复消息"""
        user_msg = x.content

        # 简单响应
        response = f"收到消息：{user_msg}"

        return Msg(self.name, response, "assistant")
```

### 3.3.2 多Agent协作

```python
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.service.service_function import function_tool

class ResearcherAgent(AgentBase):
    """研究者Agent"""

    def __init__(self, name: str):
        super().__init__(name=name)

    @function_tool
    def search_paper(self, topic: str) -> str:
        """搜索论文"""
        return f"关于{topic}的论文搜索结果"

    def reply(self, x: Msg) -> Msg:
        topic = x.content
        result = self.search_paper(topic)
        return Msg(self.name, f"研究结果：{result}", "assistant")


class WriterAgent(AgentBase):
    """写作Agent"""

    def __init__(self, name: str):
        super().__init__(name=name)

    def reply(self, x: Msg) -> Msg:
        research_result = x.content
        article = f"基于研究结果撰写的文章：{research_result[:100]}..."
        return Msg(self.name, article, "assistant")


class ManagerAgent(AgentBase):
    """管理Agent"""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.researcher = ResearcherAgent("researcher")
        self.writer = WriterAgent("writer")

    def reply(self, x: Msg) -> Msg:
        user_request = x.content

        # 协调研究
        research_msg = Msg(self.name, user_request, "to")
        research_result = self.researcher(reearch_msg)

        # 协调写作
        write_msg = Msg(self.name, research_result.content, "to")
        write_result = self.writer(write_msg)

        return write_msg  # 返回最终结果
```

## 3.4 Dify低代码平台

### 3.4.1 Dify简介

Dify是一个开源的LLM应用开发平台，提供可视化界面：

```
Dify特点：
- 可视化编排工作流
- 内置多种LLM模型
- 丰富的工具集成
- RAG能力内置
- 团队协作支持
```

### 3.4.2 Dify API集成

```python
import requests

class DifyClient:
    """Dify API客户端"""

    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_completion(self, user_input: str, app_id: str = None, inputs: dict = None):
        """创建对话"""
        url = f"{self.base_url}/v1/completions"

        data = {
            "inputs": inputs or {},
            "query": user_input,
            "response_mode": "streaming"  # 或 "blocking"
        }

        if app_id:
            url = f"{self.base_url}/v1/apps/{app_id}/invoke"

        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def chat(self, message: str, conversation_id: str = None):
        """聊天对话"""
        url = f"{self.base_url}/v1/chat-messages"

        data = {
            "inputs": {},
            "query": message,
            "response_mode": "streaming",
            "conversation_id": conversation_id
        }

        response = requests.post(url, headers=self.headers, json=data)
        return response.json()


# 使用示例
dify = DifyClient(api_key="your-api-key")

# 阻塞式回答
result = dify.create_completion("你好")
print(result)

# 流式回答
for chunk in dify.create_completion("讲个笑话", response_mode="streaming"):
    print(chunk, end="", flush=True)
```

### 3.4.3 Dify工作流配置

```yaml
# Dify工作流配置示例
workflow:
  nodes:
    - id: "start"
      type: "start"
      position: {"x": 100, "y": 100}
      config:
        inputs:
          user_query:
            type: "text"
            required: true

    - id: "llm"
      type: "llm"
      position: {"x": 300, "y": 100}
      config:
        model:
          provider: "openai"
          name: "gpt-3.5-turbo"
        inputs:
          query: "{{start.user_query}}"
        prompt: |
          请回答用户的问题：{{user_query}}

    - id: "end"
      type: "end"
      position: {"x": 500, "y": 100}
      config:
        outputs:
          result:
            type: "text"
            source: "{{llm.outputs.text}}"
```

## 3.5 本章小结

本章介绍了Agent框架的实践：
- LangChain Agent的创建和使用
- LangGraph状态图构建
- AgentScope多Agent协作
- Dify低代码平台使用

## 练习题

1. 使用LangChain创建一个带工具调用的Agent
2. 使用LangGraph构建复杂工作流
3. 设计一个多Agent协作系统
4. 使用Dify快速搭建应用原型

## 参考资料

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [AgentScope Documentation](https://agentscope.airesearch.com/)
- [Dify Documentation](https://docs.dify.ai/)
