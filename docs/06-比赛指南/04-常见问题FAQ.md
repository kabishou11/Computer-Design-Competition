# 第四章：常见问题FAQ

## 4.1 技术类问题

### Q1: 本地电脑显存不够怎么办？

**问题描述**：想运行7B模型，但电脑只有4GB显存。

**解决方案**：
1. **使用量化模型**：4-bit量化后7B模型约需6-8GB显存
2. **使用小模型**：Qwen2.5-1.5B、Phi-3.5-mini 仅需4-6GB
3. **使用云服务**：Colab、Google Cloud提供免费GPU
4. **使用API**：通义千问、Claude等API按量付费

**代码示例**（4-bit量化）：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Q2: RAG检索效果不好怎么办？

**问题描述**：检索返回的结果不相关或排序不对。

**排查步骤**：
1. 检查文本分块策略，块大小是否合适
2. 检查嵌入模型选择，中文效果是否好
3. 添加重排序模型提升精度
4. 使用混合检索（向量+关键词）

**优化建议**：
```python
# 优化后的混合检索
hybrid_results = retriever.hybrid_search(
    query=user_query,
    vector_weight=0.6,
    keyword_weight=0.4,
    top_k_final=5
)

# 添加重排序
reranked = reranker.rerank(query, [r["text"] for r in hybrid_results])
```

### Q3: 模型生成幻觉严重怎么办？

**问题描述**：模型生成的内容看起来合理但实际错误。

**解决方案**：
1. **RAG增强**：注入准确的知识库信息
2. **提示优化**：要求模型标注不确定性
3. **输出验证**：使用规则或模型验证输出
4. **降低温度**：temperature设低（如0.3）

### Q4: 对话上下文丢失怎么办？

**问题描述**：多轮对话后模型忘记了之前的对话内容。

**解决方案**：
1. **滑动窗口**：只保留最近N轮对话
2. **摘要压缩**：对历史对话进行摘要
3. **选择性保留**：只保留关键信息
4. **外部记忆**：使用向量数据库存储历史

### Q5: API调用超时或失败怎么办？

**问题描述**：调用OpenAI或其他API时经常失败。

**解决方案**：
```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
def call_api_with_retry(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            timeout=30
        )
        return response
    except Exception as e:
        print(f"API调用失败: {e}")
        raise
```

## 4.2 工具类问题

### Q6: pip安装依赖失败怎么办？

**问题描述**：安装transformers或其他包时报错。

**解决方案**：
1. 使用清华镜像源
2. 创建虚拟环境
3. 检查Python版本兼容性
4. 查看错误日志针对性解决

```bash
# 使用镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# 创建虚拟环境
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows

# 检查Python版本
python --version
```

### Q7: GPU不可用怎么办？

**问题描述**：检测不到CUDA，或者想用CPU运行。

**解决方案**：
1. 检查CUDA驱动版本
2. 安装对应版本的PyTorch
3. 设置device为"cpu"降级运行

```python
import torch

# 检查CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

# 强制使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Q8: 模型下载太慢怎么办？

**问题描述**：从HuggingFace下载模型非常慢。

**解决方案**：
1. 使用国内镜像
2. 使用模型量化版本（文件更小）
3. 使用lora_download脚本

```python
from modelscope import snapshot_download

# 使用ModelScope下载（国内快）
model_path = snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    cache_dir="./models"
)
```

## 4.3 比赛类问题

### Q9: 比赛时间不够怎么办？

**问题描述**：距离比赛截止时间很近，项目还没完成。

****快速落地策略**：
1. **缩小范围**：只实现核心功能
2. **使用现成方案**：LangChain、Dify等框架
3. **PPT优先**：即使Demo不完美，PPT要完整
4. **准备视频**：提前录制备用演示视频

### Q10: 团队成员技术不均衡怎么办？

**问题描述**：有的成员能力强，有的较弱。

**分工建议**：
- 能力强：核心算法、代码架构
- 能力弱：数据整理、文档撰写、PPT制作
- 协调机制：每日站会、代码review

### Q11: 评委问的技术问题不会怎么办？

**问题描述**：评委问到了不了解的技术点。

**应对策略**：
1. 诚实承认不足
2. 说明团队的优势方向
3. 表示后续会学习改进
4. 不要编造答案

### Q12: 项目创新性不足怎么办？

**问题描述**：担心项目没有亮点。

**提升策略**：
1. **应用场景创新**：同样的技术用在不同场景
2. **交互方式创新**：更好的用户体验设计
3. **组合创新**：将多种技术有效结合
4. **数据创新**：使用独特或高质量数据集

## 4.4 开发类问题

### Q13: 代码架构怎么设计？

**推荐项目结构**：
```
project/
├── data/           # 数据文件
├── src/
│   ├── models/     # 模型相关
│   ├── rag/        # RAG模块
│   ├── agent/      # Agent模块
│   ├── api/        # API服务
│   └── utils/      # 工具函数
├── tests/          # 测试
├── scripts/        # 脚本
├── config/         # 配置
├── requirements.txt
└── README.md
```

### Q14: 如何保证代码质量？

**最佳实践**：
1. 添加注释和文档字符串
2. 使用类型提示（Type Hints）
3. 编写单元测试
4. 使用代码检查工具（flake8、pylint）
5. 使用Git进行版本控制

### Q15: 如何处理敏感信息？

**安全做法**：
```python
# 错误做法（泄露API Key）
API_KEY = "sk-xxxxx"

# 正确做法（使用环境变量）
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# .env文件
# OPENAI_API_KEY=sk-xxxxx
```

## 4.5 学习类问题

### Q16: 如何快速学习新框架？

**学习路径**：
1. 官方文档（Quick Start）
2. 官方示例（Examples）
3. 教程视频
4. 实战项目
5. 源码阅读

### Q17: 遇到报错怎么解决？

**排查步骤**：
1. 仔细阅读错误信息
2. 搜索错误信息（Stack Overflow、GitHub Issues）
3. 查看官方FAQ
4. 在社区提问
5. 阅读源码

### Q18: 如何持续提升？

**建议**：
1. 定期阅读论文（arXiv）
2. 关注技术博客（Medium、知乎）
3. 参与开源项目
4. 复现经典论文
5. 写技术博客总结

## 4.6 资源推荐

### 4.6.1 学习资源

| 资源类型 | 推荐 |
|---------|------|
| **入门教程** | HuggingFace官方教程 |
| **进阶课程** | 李宏毅机器学习/深度学习 |
| **论文** | arXiv AI专区 |
| **社区** | GitHub、知乎、CSDN |

### 4.6.2 工具资源

| 类型 | 推荐 |
|-----|------|
| **模型** | HuggingFace、ModelScope |
| **数据集** | HuggingFace Datasets |
| **论文** | Papers With Code |
| **工具** | LangChain、LlamaIndex |

## 4.7 本章小结

本章收集了常见问题及解决方案：
- 技术类问题（显存、幻觉、API等）
- 工具类问题（安装、GPU、下载等）
- 比赛类问题（时间、创新、答辩等）
- 开发类问题（架构、安全、质量等）
- 学习类问题（学习路径、问题解决等）

建议遇到问题时先查阅本章，如果没有找到答案再寻求帮助。
