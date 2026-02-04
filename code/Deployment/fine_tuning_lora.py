"""
高效微调示例
使用PEFT进行LoRA微调
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_int8_training
)
from datasets import Dataset
from typing import List, Dict


# ==================== 配置 ====================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./output"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
EPOCHS = 3
MAX_LENGTH = 1024


# ==================== 数据准备 ====================

def create_instruction_dataset(data: List[Dict]) -> Dataset:
    """创建指令微调数据集"""
    formatted_data = []

    for item in data:
        # 构造对话
        messages = []

        if "system" in item:
            messages.append({"role": "system", "content": item["system"]})

        messages.append({"role": "user", "content": item["instruction"]})
        messages.append({"role": "assistant", "content": item["output"]})

        formatted_data.append({"messages": messages})

    return Dataset.from_list(formatted_data)


def preprocess_function(examples, tokenizer, max_length=MAX_LENGTH):
    """预处理函数"""
    messages_list = examples["messages"]

    # 使用chat template
    texts = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # Tokenize
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 设置labels
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs


# ==================== 模型准备 ====================

def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer


def prepare_lora_model(model):
    """准备LoRA模型"""
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    model.print_trainable_parameters()

    return model


# ==================== 训练 ====================

def train_model(model, tokenizer, train_dataset, eval_dataset=None):
    """训练模型"""
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluation_strategy="epoch" if eval_dataset else "no",
    )

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存模型
    logger.info("保存模型...")
    trainer.save_model(OUTPUT_DIR)

    return model


# ==================== 推理 ====================

def merge_and_export(model, tokenizer, output_path="./output/merged"):
    """合并LoRA权重并导出"""
    from peft import PeftModel

    # 如果模型是PeftModel，直接merge
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    # 保存
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"模型已保存到 {output_path}")


# ==================== 主程序 ====================

def main():
    """主程序"""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. 准备数据（示例数据）
    train_data = [
        {
            "instruction": "什么是人工智能？",
            "output": "人工智能（AI）是计算机科学的一个重要分支，它致力于创建能够模拟人类智能的系统，包括学习、推理、问题解决等能力。"
        },
        {
            "instruction": "解释一下机器学习。",
            "output": "机器学习是人工智能的一个子领域，它让计算机通过数据和算法自动学习和改进，而不需要明确的编程指令。"
        },
        # ... 更多训练数据
    ]

    logger.info("准备数据集...")
    train_dataset = create_instruction_dataset(train_data)

    # 2. 加载模型
    logger.info(f"加载模型 {MODEL_NAME}...")
    model, tokenizer = load_model_and_tokenizer()

    # 3. 预处理数据
    logger.info("预处理数据...")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["messages"]
    )

    # 4. 准备LoRA
    logger.info("准备LoRA...")
    model = prepare_lora_model(model)

    # 5. 训练
    logger.info("开始训练...")
    model = train_model(model, tokenizer, train_dataset)

    # 6. 导出
    logger.info("导出模型...")
    merge_and_export(model, tokenizer)

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
