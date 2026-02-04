"""
本地模型推理示例
使用Transformers进行本地推理
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Dict, Optional


class LocalLLM:
    """本地LLM推理类"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "float16"
    ):
        """
        Args:
            model_name: 模型名称或本地路径
            device: 设备 ("cuda", "cpu", "auto")
            torch_dtype: 精度 ("float16", "bfloat16", "float32")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 设置精度
        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float16

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )

        print(f"模型已加载到 {self.device}")
        print(f"模型参数量: {self.model.num_parameters() / 1e9:.2f}B")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        stop_strings: List[str] = None
    ) -> str:
        """
        生成文本

        Args:
            prompt: 提示词
            max_tokens: 最大生成长度
            temperature: 温度（越高越随机）
            top_p: 核采样参数
            top_k: Top-K采样
            stream: 是否流式输出
            stop_strings: 停止字符串

        Returns:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if stream:
            return self._stream_generate(inputs, max_tokens, temperature, top_p, top_k, stop_strings)
        else:
            return self._batch_generate(inputs, max_tokens, temperature, top_p, top_k, stop_strings)

    def _batch_generate(
        self,
        inputs,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_strings: List[str]
    ) -> str:
        """批量生成"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码（去掉输入部分）
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return self._truncate_stop_strings(response, stop_strings)

    def _stream_generate(
        self,
        inputs,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_strings: List[str]
    ) -> str:
        """流式生成"""
        from threading import Thread

        generated_text = []
        stop_flag = [False]

        def generate():
            with torch.no_grad():
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer
                )

        # 启动生成线程
        thread = Thread(target=generate)
        thread.start()
        thread.join()

        return "[已流式输出]"

    def _truncate_stop_strings(self, text: str, stop_strings: List[str]) -> str:
        """截断停止字符串"""
        if not stop_strings:
            return text

        for stop in stop_strings:
            if stop in text:
                text = text[:text.index(stop)]

        return text

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        对话模式

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成长度
            temperature: 温度
            **kwargs: 其他参数
        """
        # 转换为对话格式
        prompt = self._format_messages(messages)

        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息"""
        # 检查是否支持chat template
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # 手动格式化
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"

        return text

    def count_tokens(self, text: str) -> int:
        """计算token数"""
        return len(self.tokenizer.encode(text))

    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "cached": torch.cuda.memory_reserved() / 1024**3
            }
        return {"message": "CUDA not available"}


def demo():
    """演示"""
    print("=" * 60)
    print("本地LLM推理演示")
    print("=" * 60)

    # 初始化模型（根据硬件选择）
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或本地路径

    print("\n正在加载模型...")
    llm = LocalLLM(model_name)

    # 1. 简单生成
    print("\n" + "=" * 60)
    print("示例1: 简单文本生成")
    print("=" * 60)

    prompt = "介绍一下人工智能的发展历程。"
    response = llm.generate(prompt, max_tokens=200, temperature=0.7)
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")

    # 2. 对话模式
    print("\n" + "=" * 60)
    print("示例2: 对话模式")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是..."},
        {"role": "user", "content": "它和深度学习有什么区别？"}
    ]

    response = llm.chat(messages, max_tokens=300, temperature=0.7)
    print("\n对话历史:")
    for m in messages:
        print(f"  {m['role']}: {m['content'][:50]}...")
    print(f"\nResponse:\n{response}")

    # 内存使用
    print("\n" + "=" * 60)
    print("内存使用情况")
    print("=" * 60)
    print(llm.get_memory_usage())


if __name__ == "__main__":
    demo()
