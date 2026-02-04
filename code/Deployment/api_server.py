"""
FastAPI服务示例
将LLM封装为REST API服务
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import uvicorn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 数据模型 ====================

class GenerateRequest(BaseModel):
    """文本生成请求"""
    prompt: str = Field(..., description="输入提示词")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p参数")
    stream: bool = Field(default=False, description="是否流式输出")


class GenerateResponse(BaseModel):
    """文本生成响应"""
    text: str
    usage: Dict = {}


class Message(BaseModel):
    """对话消息"""
    role: str = Field(..., description="角色 (user/assistant/system)")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""
    messages: List[Message] = Field(..., description="对话历史")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """对话响应"""
    response: str
    usage: Dict = {}


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool


# ==================== 全局变量 ====================

llm_model = None
tokenizer = None


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global llm_model, tokenizer

    # 启动时加载模型
    logger.info("正在加载模型...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("模型加载完成")
        yield

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    # 关闭时清理
    logger.info("服务关闭")


# ==================== FastAPI应用 ====================

app = FastAPI(
    title="LLM API Server",
    description="大语言模型推理服务",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== API端点 ====================

@app.get("/", summary="根路径")
async def root():
    """根路径"""
    return {
        "service": "LLM API Server",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health():
    """检查服务健康状态"""
    return HealthResponse(
        status="ok",
        model_loaded=llm_model is not None
    )


@app.post("/generate", response_model=GenerateResponse, summary="文本生成")
async def generate(request: GenerateRequest):
    """
    文本生成接口

    发送提示词，返回生成的文本
    """
    if llm_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        import torch

        # 编码输入
        inputs = tokenizer(request.prompt, return_tensors="pt").to(llm_model.device)

        # 生成
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        return GenerateResponse(
            text=text,
            usage={"input_tokens": input_length, "output_tokens": len(generated)}
        )

    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="对话")
async def chat(request: ChatRequest):
    """
    对话接口

    发送对话历史，返回助手回复
    """
    if llm_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        import torch

        # 转换为对话格式
        messages = [m.dict() for m in request.messages]

        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant: "

        # 编码
        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

        # 生成
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[0][input_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        return ChatResponse(
            response=text,
            usage={"input_tokens": input_length, "output_tokens": len(generated)}
        )

    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 主程序 ====================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
