# 第四章：大语言模型训练策略

## 4.1 预训练到应用的两阶段

大语言模型从预训练到实际应用通常需要两个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   阶段1: 预训练 (Pre-training)                              │
│   ├── 目标: 学习通用语言知识和世界知识                        │
│   ├── 数据: 海量无标注文本 (万亿token)                        │
│   └── 成本: 极高（需要数千GPU）                              │
│                                                             │
│   阶段2: 微调 (Fine-tuning)                                 │
│   ├── 目标: 适配特定任务或领域                                │
│   ├── 数据: 少量标注数据 (数千-数万条)                        │
│   └── 成本: 相对较低                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 微调方法概览

| 方法 | 全量参数 | 额外参数 | 显存需求 | 适用场景 |
|-----|---------|---------|---------|---------|
| **全量微调** | ✓ | - | 极高 | 充足算力 |
| **LoRA** | - | ✓ | 中等 | 推荐首选 |
| **QLoRA** | - | ✓ | 低 | 单卡可训练 |
| **Prefix Tuning** | - | ✓ | 低 | 快速实验 |
| **Adapter** | - | ✓ | 中等 | 迁移学习 |

## 4.3 LoRA微调详解

### 4.3.1 LoRA原理

LoRA（Low-Rank Adaptation）通过在权重矩阵旁添加低秩分解的 adapter 层来微调模型：

```
原始权重: W ∈ R^(d×k)
更新: ΔW = BA，其中 B ∈ R^(d×r), A ∈ R^(r×k), r << d,k

训练时: 前向 = Wx + BAx
        梯度更新 ΔW = ∂L/∂W = (∂L/∂x) · B · A
冻结W，只训练B和A
```

### 4.3.2 LoRA实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """LoRA适配器层"""
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # A: 降维矩阵，初始化为随机值
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: 升维矩阵，初始化为零
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.lora_A, mean=0, std=0.02)

    def forward(self, x, W):
        # 原始前向: W @ x.T
        # LoRA前向: W @ x + (alpha/rank) * B @ A @ x
        return F.linear(x, W, bias=None) + (self.alpha / self.rank) * F.linear(
            F.linear(x, self.lora_A), self.lora_B
        )

class LoRALinear(nn.Module):
    """带LoRA的线性层"""
    def __init__(self, linear_layer, rank=16, alpha=32):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )
        # 冻结原始权重
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.lora(x, self.linear.weight)

def replace_linear_with_lora(model, target_modules=["q_proj", "v_proj"], rank=16):
    """将模型中的指定线性层替换为LoRA版本"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                setattr(model, name, LoRALinear(module, rank))
    return model
```

### 4.3.2 使用PEFT库实现LoRA

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 添加LoRA适配器
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 6,291,456 || all params: 7,634,923,520 || trainable%: 0.0824
```

## 4.4 QLoRA微调

### 4.4.1 QLoRA原理

QLoRA在LoRA基础上增加了量化技术，进一步降低显存需求：

- **4-bit NormalFloat**: 高效的量化格式
- **双重量化**: 量化后的量化（量化常数）
- **分页优化器**: 利用NVIDIA统一管理优化器状态

### 4.4.2 QLoRA实现

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_int8_training

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 双重量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备训练
model = prepare_model_for_int8_training(model)

# 添加LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

## 4.5 SFT（监督微调）

### 4.5.1 数据准备

```python
from datasets import Dataset
import json

# 格式化为对话数据
train_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的法律助手。"},
            {"role": "user", "content": "什么是合同法？"},
            {"role": "assistant", "content": "合同法是调整平等主体之间..."
            }
        ]
    },
    # ... 更多数据
]

# 转换为训练格式
def format_data(sample):
    text = ""
    for msg in sample["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

formatted_data = [format_data(d) for d in train_data]
dataset = Dataset.from_list(formatted_data)

# 使用tokenizer处理
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 4.5.2 训练配置

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 4.6 RLHF（基于人类反馈的强化学习）

### 4.6.1 RLHF三阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Stage 1: SFT (监督微调)                                        │
│  └── 使用标注数据微调模型                                        │
│                                                                 │
│  Stage 2: 奖励模型训练                                          │
│  └── 收集人类偏好数据，训练奖励模型(RM)                          │
│                                                                 │
│  Stage 3: PPO强化训练                                          │
│  └── 使用PPO算法，根据RM信号优化策略                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6.2 奖励模型训练

```python
import torch.nn as nn

class RewardModel(nn.Module):
    """奖励模型"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        # 使用最后一个token的表示作为奖励信号
        reward = self.reward_head(hidden_states[:, -1, :])
        return reward

def compute_reward_loss(rewards_chosen, rewards_rejected):
    """计算PPO风格的奖励损失"""
    # 奖励差距损失
    loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    return loss
```

### 4.6.3 PPO训练

```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=2,
    ppo_epochs=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=data_collator,
)

for epoch in range(ppo_config.num_epochs):
    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]
        response_tensors = generate_responses(model, query_tensors)

        # 计算奖励
        texts = tokenizer.batch_decode(response_tensors)
        rewards = [reward_model.get_reward(text) for text in texts]

        # PPO更新
        ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## 4.7 高效训练技巧

### 4.7.1 DeepSpeed ZeRO

```python
from accelerate import Accelerator
from deepspeed import DeepSpeedConfig, DeepSpeedPlugin

# DeepSpeed配置
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
    },
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
}

# 使用Accelerator
accelerator = Accelerator(
    deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config)
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

### 4.7.2 Flash Attention

```python
from flash_attn import FlashAttentionLayer

# 使用Flash Attention替代标准注意力
model.layers[0].self_attn = FlashAttentionLayer(
    embed_dim=4096,
    num_heads=32,
    attention_dropout=0.0,
    causal=True,
)
```

### 4.7.3 梯度检查点

```python
# 激活梯度检查点以节省显存
model.gradient_checkpointing_enable()

# 或者只对部分层启用
model.apply(model._set_gradient_checkpointing)
```

## 4.8 本章小结

本章介绍了几种主流的大语言模型微调方法：
- **LoRA**: 低秩适配，显存效率高
- **QLoRA**: 4-bit量化，单卡可训练
- **SFT**: 监督微调，对话数据格式
- **RLHF**: 基于人类反馈，需要奖励模型

对于比赛项目，推荐使用LoRA或QLoRA进行微调，资源消耗低且效果良好。

## 练习题

1. 使用LoRA微调一个7B模型，训练一个特定领域的助手
2. 比较全量微调和LoRA的显存占用差异
3. 构建一个高质量的SFT数据集
4. 了解RLHF的实现细节和训练技巧

## 参考资料

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - LoRA原始论文
- [QLoRA Paper](https://arxiv.org/abs/2305.14389) - QLoRA原始论文
- [PEFT Library](https://github.com/huggingface/peft) - Hugging Face PEFT
- [TRL Library](https://github.com/huggingface/trl) - RLHF训练工具
