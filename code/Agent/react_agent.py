"""
ReAct Agent实现
基于推理-行动-观察循环的智能体
"""

from typing import List, Dict, Tuple, Callable, Optional
import json


class ReActAgent:
    """ReAct Agent实现"""

    def __init__(
        self,
        llm_callable: Callable,
        tools: Dict[str, Dict] = None,
        max_iterations: int = 10
    ):
        """
        Args:
            llm_callable: LLM调用函数
            tools: 工具字典 {name: {"function": func, "description": desc}}
            max_iterations: 最大迭代次数
        """
        self.llm = llm_callable
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.history = []

    def add_tool(self, name: str, func: Callable, description: str):
        """添加工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def run(self, query: str) -> Tuple[str, List[Dict]]:
        """
        运行Agent

        Returns:
            Tuple[最终答案, 推理历史]
        """
        context = ""
        history = []

        for iteration in range(self.max_iterations):
            # 构建思考提示
            prompt = self._build_thought_prompt(query, context, history)

            # 获取思考和行动
            response = self.llm(prompt)

            # 解析输出
            thought, action, action_input = self._parse_response(response)

            # 记录步骤
            step_info = {
                "iteration": iteration + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input
            }
            history.append(step_info)

            # 执行行动或结束
            if action.upper() == "FINISH":
                return action_input or response.split("Answer:")[-1].strip(), history

            if action and action in self.tools:
                # 执行工具
                result = self.tools[action]["function"](action_input)
                context += f"\n观察: {result}"
                step_info["observation"] = result
            else:
                # 无法执行行动，尝试直接回答
                context += f"\n观察: 无法识别工具 '{action}'"
                step_info["observation"] = f"Unknown tool: {action}"

        # 达到最大迭代次数
        final_answer = self._finalize_answer(query, history)
        return final_answer, history

    def _build_thought_prompt(
        self,
        query: str,
        context: str,
        history: List[Dict]
    ) -> str:
        """构建思考提示"""
        tool_descs = []
        for name, info in self.tools.items():
            tool_descs.append(f"- {name}: {info['description']}")

        history_text = ""
        for step in history:
            history_text += f"\n迭代{step['iteration']}:\n"
            history_text += f"  思考: {step['thought']}\n"
            history_text += f"  行动: {step['action']}[{step['action_input']}]\n"
            if 'observation' in step:
                history_text += f"  观察: {step['observation']}\n"

        prompt = f"""
你是一个智能助手，通过思考和行动来解决问题。

可用工具：
{chr(10).join(tool_descs)}

历史步骤：
{history_text}

当前上下文：{context}

请按照以下格式继续思考和行动：
Thought: [描述你的推理过程]
Action: [要执行的工具名称，或 FINISH 表示完成]
Action Input: [工具的参数，如果 action 是 FINISH，这里放最终答案]
Answer: [仅当 action 是 FINISH 时提供最终答案]

用户问题：{query}

开始思考：
"""
        return prompt

    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """解析LLM响应"""
        lines = response.strip().split('\n')

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

        return thought, action, action_input

    def _finalize_answer(self, query: str, history: List[Dict]) -> str:
        """生成最终答案"""
        history_summary = "\n".join([
            f"步骤{i+1}: {step['thought']} -> {step.get('action', 'N/A')}"
            for i, step in enumerate(history)
        ])

        prompt = f"""
根据以下推理历史，回答用户问题。

推理历史：
{history_summary}

用户问题：{query}

请基于以上推理，给出最终答案：
"""
        return self.llm(prompt)


def demo():
    """演示"""

    # 模拟LLM
    def mock_llm(prompt):
        # 简单模拟：根据关键词返回
        if "搜索" in prompt or "人工智能" in prompt:
            return """Thought: 用户询问关于人工智能的问题，我应该先搜索相关信息。
Action: search
Action Input: 人工智能的定义和应用

Answer: 人工智能是计算机科学的重要分支，致力于创建能够模拟人类智能的系统。"""

        if "计算" in prompt or "算术" in prompt:
            return """Thought: 用户要求计算，我应该使用计算器。
Action: calculate
Action Input: 25 * 4 + 10

Answer: 25 * 4 + 10 = 110"""

        return """Thought: 这是一个直接的问题，我可以直接回答。
Action: FINISH
Action Input: 您好！我是ReAct智能助手，可以帮助您回答问题和执行任务。"""

    # 定义工具
    def search_knowledge(query: str) -> str:
        knowledge = {
            "人工智能": "人工智能(AI)是计算机科学分支，研究创建智能机器",
            "机器学习": "机器学习是AI子领域，让计算机通过数据学习",
            "深度学习": "深度学习是ML分支，使用多层神经网络"
        }
        for key, value in knowledge.items():
            if key in query:
                return value
        return f"搜索'{query}'的结果"

    def calculate(expr: str) -> str:
        try:
            result = eval(expr)
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{e}"

    def get_time() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 创建Agent
    agent = ReActAgent(mock_llm, max_iterations=5)
    agent.add_tool("search", search_knowledge, "搜索知识库")
    agent.add_tool("calculate", calculate, "执行数学计算")
    agent.add_tool("time", get_time, "获取当前时间")

    # 运行
    print("=" * 60)
    print("ReAct Agent 演示")
    print("=" * 60)

    questions = [
        "什么是人工智能？",
        "计算一下 100 / 5 + 20",
        "现在几点了？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 40)

        answer, history = agent.run(q)

        print("推理过程:")
        for step in history:
            print(f"  步骤 {step['iteration']}:")
            print(f"    思考: {step['thought']}")
            print(f"    行动: {step['action']}[{step['action_input']}]")
            if 'observation' in step:
                print(f"    观察: {step['observation']}")

        print(f"\n最终答案: {answer}")
        print("=" * 60)


if __name__ == "__main__":
    demo()
