+++
date = '2024-09-12T16:45:00+08:00'
draft = true
title = 'AI 模型微调（Fine-tuning）完全指南'
image = '/images/covers/fine-tuning.jpg'
categories = ['AI 开发']
tags = ['Fine-tuning', '模型训练', '机器学习']
+++

模型微调是让预训练模型适应特定任务的关键技术。本文详细介绍 AI 模型微调的原理、方法和实践。

## 什么是微调

微调（Fine-tuning）是在预训练模型的基础上，使用特定领域的数据进行进一步训练，使模型更好地适应特定任务。

### 为什么需要微调

1. **提升特定任务表现**：针对性优化
2. **减少训练成本**：基于已有知识
3. **数据需求更少**：相比从头训练
4. **保留通用能力**：不失去预训练知识

## 微调方法

### 1. 全量微调（Full Fine-tuning）

更新模型所有参数：

```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 所有参数都可训练
for param in model.parameters():
    param.requires_grad = True

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### 2. 参数高效微调（PEFT）

#### LoRA（Low-Rank Adaptation）

只训练少量参数：

````python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 110,104,320 || trainable%: 0.27


#### Adapter

插入小型适配器层：

```python
from transformers.adapters import AdapterConfig

config = AdapterConfig.load("pfeiffer")
model.add_adapter("task_adapter", config=config)
model.train_adapter("task_adapter")


### 3. 提示微调（Prompt Tuning）

只训练提示参数：

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment:"
)

model = get_peft_model(model, config)


## 数据准备

### 1. 数据格式

```python
# 分类任务
train_data = [
    {"text": "这个产品很好用", "label": 1},
    {"text": "质量太差了", "label": 0},
    # ...
]

# 转换为Dataset
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": [d["text"] for d in train_data],
    "label": [d["label"] for d in train_data]
})


### 2. 数据增强

```python
import nlpaug.augmenter.word as naw

# 同义词替换
aug = naw.SynonymAug(aug_src='wordnet')
augmented = aug.augment(text)

# 回译
aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-zh-en',
    to_model_name='Helsinki-NLP/opus-mt-en-zh'
)
augmented = aug.augment(text)


### 3. 数据清洗

```python
def clean_text(text):
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 移除多余空格
    text = ' '.join(text.split())
    # 转小写（如果需要）
    text = text.lower()
    return text

dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})


## 训练配置

### 超参数设置

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


### 学习率调度

```python
from transformers import get_scheduler

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


## 评估与优化

### 1. 评估指标

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


### 2. 早停

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    # ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


### 3. 混淆矩阵

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

cm = confusion_matrix(test_dataset['label'], preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


## 实战案例：情感分类微调

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. 加载数据
dataset = load_dataset('csv', data_files={
    'train': 'train.csv',
    'test': 'test.csv'
})

# 2. 初始化
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 5. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# 6. 保存模型
trainer.save_model("./sentiment_model_final")


## 最佳实践

1. **从小规模开始**：先用小数据集验证
2. **监控过拟合**：使用验证集，观察 loss 曲线
3. **学习率调优**：从小开始，逐步调整
4. **数据质量**：确保标注准确性
5. **定期评估**：在验证集上持续监控
6. **版本管理**：记录每次实验的配置和结果

## 常见问题

### 过拟合

- 增加数据
- 使用 Dropout
- 减小模型复杂度
- Early Stopping

### 训练不收敛

- 降低学习率
- 增加 warmup 步数
- 检查数据质量
- 尝试不同优化器

### 显存不足

- 减小 batch size
- 使用梯度累积
- 使用混合精度训练
- 尝试 PEFT 方法

## 总结

模型微调是 AI 应用落地的关键技术。选择合适的微调方法，准备高质量的数据，设置合理的超参数，就能获得优秀的模型表现。记住：没有万能的配置，需要根据具体任务不断实验和调优。
````
