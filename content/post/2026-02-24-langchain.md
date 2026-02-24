+++
date = '2026-02-24T23:39:04+08:00'
draft = false
title = 'LangChain 深入解析：从零开始的完整指南'
image = '/images/covers/langchain.jpg'
categories = ['AI', 'Agent']
tags = ['AI', 'LangChain']
+++

# LangChain 深入解析：从零开始的完整指南

---

## 目录

- [一、LangChain 是什么？](#一langchain-是什么)
- [二、为什么需要 LangChain？](#二为什么需要-langchain)
- [三、整体架构](#三整体架构)
- [四、核心概念与组件](#四核心概念与组件)
  - [4.1 Model I/O（模型输入/输出）](#41-model-io模型输入输出)
  - [4.2 Prompt Templates（提示词模板）](#42-prompt-templates提示词模板)
  - [4.3 Output Parsers（输出解析器）](#43-output-parsers输出解析器)
  - [4.4 Chains（链）](#44-chains链)
  - [4.5 Memory（记忆）](#45-memory记忆)
  - [4.6 Agents（智能体）](#46-agents智能体)
  - [4.7 Tools（工具）](#47-tools工具)
  - [4.8 Retrieval / RAG](#48-retrieval--rag)
  - [4.9 Callbacks（回调）](#49-callbacks回调)
- [五、LCEL — LangChain 表达式语言](#五lcel--langchain-表达式语言)
- [六、实战案例](#六实战案例)
- [七、LangChain 生态系统](#七langchain-生态系统)
- [八、最佳实践与常见陷阱](#八最佳实践与常见陷阱)
- [九、总结与学习路线](#九总结与学习路线)

---

## 一、LangChain 是什么？

### 1.1 定义

**LangChain** 是一个用于开发 **由大语言模型（LLM）驱动的应用程序** 的开源框架。它由 Harrison Chase 于 2022 年 10 月发布，最初为 Python 库，后扩展至 JavaScript/TypeScript。

> 一句话概括：LangChain 把 LLM 的能力 **"链接"** 起来，让开发者能快速构建复杂的 AI 应用。

### 1.2 核心理念

```
┌─────────────────────────────────────────────┐
│              LangChain 核心理念              │
├─────────────────────────────────────────────┤
│  1. 组件化 (Composability)                   │
│     → 每个功能都是可插拔的模块               │
│                                             │
│  2. 数据感知 (Data-aware)                    │
│     → LLM 能连接外部数据源                   │
│                                             │
│  3. 代理能力 (Agentic)                       │
│     → LLM 能自主决策、调用工具               │
└─────────────────────────────────────────────┘
```

### 1.3 版本演进

| 时间      | 版本/事件 | 重要变化                           |
| --------- | --------- | ---------------------------------- |
| 2022.10   | 首次发布  | 基础Chain概念                      |
| 2023.Q1   | 快速迭代  | Agent、Memory成熟                  |
| 2023.08   | LCEL发布  | 表达式语言，声明式编排             |
| 2023.10   | LangServe | 一键部署为API                      |
| 2024.01   | v0.1.0    | 稳定API，拆分包结构                |
| 2024.Q2   | v0.2.x    | `langchain-core` 独立，弃用旧Chain |
| 2024-2025 | v0.3.x    | 全面拥抱LCEL，LangGraph成熟        |

---

## 二、为什么需要 LangChain？

### 2.1 直接调用 API 的痛点

```python
# 直接使用 OpenAI API — 简单场景没问题
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
```

但当需求变复杂时：

```
❌ 需要多轮对话记忆 → 自己管理上下文
❌ 需要查询数据库   → 自己写胶水代码
❌ 需要搜索互联网   → 自己对接搜索API
❌ 需要解析结构化输出 → 自己写解析逻辑
❌ 需要链式多步推理  → 自己编排流程
❌ 需要切换模型供应商 → 重写所有代码
```

### 2.2 LangChain 解决了什么

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   统一接口    │     │   组件编排    │     │   生态集成    │
│              │     │              │     │              │
│ OpenAI       │     │ Chain        │     │ 向量数据库    │
│ Anthropic    │     │ Agent        │     │ 搜索引擎      │
│ HuggingFace  │────▶│ Memory       │────▶│ 数据库        │
│ 本地模型     │     │ RAG Pipeline │     │ API服务       │
│ ...          │     │ ...          │     │ ...           │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 三、整体架构

### 3.1 包结构（v0.2+）

```
langchain 生态系统
│
├── langchain-core          # 🔧 核心抽象层（接口定义、LCEL、基础类型）
│   ├── BaseLanguageModel
│   ├── BaseChatModel
│   ├── BaseRetriever
│   ├── BaseOutputParser
│   ├── RunnableInterface    # LCEL核心
│   └── ...
│
├── langchain                # 🏗️ 编排层（Chains, Agents, 高级逻辑）
│   ├── chains/
│   ├── agents/
│   ├── memory/
│   └── ...
│
├── langchain-community      # 🌐 第三方集成（社区维护）
│   ├── chat_models/
│   ├── vectorstores/
│   ├── document_loaders/
│   └── ...
│
├── langchain-openai         # 📦 官方合作集成包
├── langchain-anthropic
├── langchain-google-genai
├── langchain-...
│
├── langgraph               # 🔄 有状态多Agent编排（图结构）
├── langserve               # 🚀 部署为REST API
└── langsmith               # 📊 监控、调试、评估平台
```

### 3.2 核心架构图

```
                          用户输入 (User Input)
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Prompt Template    │  ← 构造提示词
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LLM / Chat      │  ← 调用模型
                    │      Model          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Output Parser     │  ← 解析输出
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
    ┌───────────┐      ┌───────────┐       ┌───────────┐
    │  Memory   │      │  Tools    │       │ Retriever │
    │  (记忆)   │      │  (工具)   │       │ (检索器)  │
    └───────────┘      └───────────┘       └───────────┘
```

---

## 四、核心概念与组件

### 4.1 Model I/O（模型输入/输出）

这是 LangChain 最基础的部分——与大语言模型交互。

#### 4.1.1 两种模型类型

```python
# ① LLM（纯文本补全模型）—— 输入文本，输出文本
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("天空为什么是蓝色的？")
print(result)  # 返回 str

# ② ChatModel（对话模型）—— 输入消息列表，输出消息 ⭐推荐
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4o")
```

#### 4.1.2 消息类型

```python
from langchain_core.messages import (
    SystemMessage,    # 系统指令 → 设定角色/行为
    HumanMessage,     # 用户消息
    AIMessage,        # AI回复
    ToolMessage,      # 工具调用结果
    FunctionMessage,  # (旧版) 函数调用结果
)

messages = [
    SystemMessage(content="你是一位专业的Python开发者"),
    HumanMessage(content="解释一下装饰器是什么"),
]

response = chat_model.invoke(messages)
print(response.content)        # 文本内容
print(response.response_metadata)  # token用量等元信息
```

#### 4.1.3 模型通用参数

```python
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,       # 创造性 (0=确定性, 2=最随机)
    max_tokens=1000,       # 最大输出token数
    timeout=30,            # 超时秒数
    max_retries=2,         # 重试次数
    api_key="sk-...",      # API密钥（建议用环境变量）
    base_url="https://...", # 自定义端点（用于代理/私有部署）
)
```

#### 4.1.4 流式输出

```python
# 流式输出 —— 逐token返回，提升用户体验
for chunk in chat_model.stream("讲一个笑话"):
    print(chunk.content, end="", flush=True)

# 异步流式
async for chunk in chat_model.astream("讲一个笑话"):
    print(chunk.content, end="", flush=True)
```

#### 4.1.5 多模型切换

```python
# LangChain 的最大优势之一：统一接口，自由切换模型

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 三个不同的模型，完全相同的调用方式
models = {
    "openai": ChatOpenAI(model="gpt-4o"),
    "claude": ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    "gemini": ChatGoogleGenerativeAI(model="gemini-pro"),
}

for name, model in models.items():
    response = model.invoke("你好，你是谁？")
    print(f"{name}: {response.content}")
```

---

### 4.2 Prompt Templates（提示词模板）

提示词工程的核心工具——将动态变量注入到精心设计的提示词中。

#### 4.2.1 基础字符串模板

```python
from langchain_core.prompts import PromptTemplate

# 方式一：显式指定变量
prompt = PromptTemplate(
    input_variables=["product"],
    template="请为{product}写一句广告语"
)

# 方式二：自动推断变量（推荐）
prompt = PromptTemplate.from_template(
    "请为{product}写一句广告语，风格要{style}"
)

# 格式化
result = prompt.format(product="运动鞋", style="年轻活力")
print(result)  # "请为运动鞋写一句广告语，风格要年轻活力"

# 也可以用 invoke
result = prompt.invoke({"product": "运动鞋", "style": "年轻活力"})
# 返回 StringPromptValue 对象
```

#### 4.2.2 聊天消息模板 ⭐

```python
from langchain_core.prompts import ChatPromptTemplate

# 最常用的方式
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}专家，请用{language}回答"),
    ("human", "{question}"),
])

# 格式化为消息列表
messages = prompt.format_messages(
    role="机器学习",
    language="中文",
    question="什么是梯度下降？"
)

# 直接与模型串联
chain = prompt | chat_model
response = chain.invoke({
    "role": "机器学习",
    "language": "中文",
    "question": "什么是梯度下降？"
})
```

#### 4.2.3 MessagesPlaceholder（历史消息占位符）

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="chat_history"),  # 插入历史对话
    ("human", "{input}"),
])

from langchain_core.messages import HumanMessage, AIMessage

messages = prompt.format_messages(
    chat_history=[
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好小明！有什么可以帮你的？"),
    ],
    input="我叫什么名字？"
)
```

#### 4.2.4 FewShotPromptTemplate（少样本提示）

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "开心", "output": "😊"},
    {"input": "难过", "output": "😢"},
    {"input": "生气", "output": "😠"},
]

# 单个示例的格式
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入: {input}\n输出: {output}"
)

# 组装少样本模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="将情绪转换为表情符号：",
    suffix="输入: {input}\n输出:",
    input_variables=["input"],
)

print(few_shot_prompt.format(input="惊讶"))
```

输出：

```
将情绪转换为表情符号：

输入: 开心
输出: 😊

输入: 难过
输出: 😢

输入: 生气
输出: 😠

输入: 惊讶
输出:
```

---

### 4.3 Output Parsers（输出解析器）

让 LLM 的文本输出转化为 **结构化数据**。

#### 4.3.1 常用解析器对比

| 解析器                           | 输出类型       | 适用场景       |
| -------------------------------- | -------------- | -------------- |
| `StrOutputParser`                | `str`          | 简单文本输出   |
| `JsonOutputParser`               | `dict`         | JSON结构化数据 |
| `PydanticOutputParser`           | Pydantic Model | 强类型校验     |
| `CommaSeparatedListOutputParser` | `List[str]`    | 逗号分隔列表   |
| `StructuredOutputParser`         | `dict`         | 自定义字段     |
| `XMLOutputParser`                | `dict`         | XML格式        |

#### 4.3.2 StrOutputParser

```python
from langchain_core.output_parsers import StrOutputParser

# 最简单——提取 AIMessage 中的文本
chain = prompt | chat_model | StrOutputParser()
result = chain.invoke({"question": "什么是AI？"})
print(type(result))  # <class 'str'>
```

#### 4.3.3 PydanticOutputParser ⭐

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 定义数据结构
class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，1-10分")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    summary: str = Field(description="一句话总结")

# 创建解析器
parser = PydanticOutputParser(pydantic_object=MovieReview)

# 获取格式指令（会自动注入到提示词中）
print(parser.get_format_instructions())

# 完整链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个电影评论家。{format_instructions}"),
    ("human", "请评价电影《{movie}》"),
])

chain = prompt | chat_model | parser

review = chain.invoke({
    "movie": "盗梦空间",
    "format_instructions": parser.get_format_instructions()
})

print(review.title)    # "盗梦空间"
print(review.rating)   # 9.2
print(review.pros)     # ["创意独特", "视觉震撼", ...]
```

#### 4.3.4 with_structured_output（v0.2+ 推荐方式）⭐

```python
# 更简洁的方式——直接让模型输出结构化数据
from pydantic import BaseModel, Field

class Joke(BaseModel):
    """一个笑话"""
    setup: str = Field(description="笑话的铺垫")
    punchline: str = Field(description="笑话的包袱")

# 直接绑定到模型
structured_llm = chat_model.with_structured_output(Joke)
result = structured_llm.invoke("讲一个关于程序员的笑话")

print(result.setup)      # str
print(result.punchline)  # str
# 底层使用 function calling / tool calling 实现，更可靠
```

---

### 4.4 Chains（链）

**Chain** 是 LangChain 的核心概念——将多个组件串联成一个完整的处理流程。

#### 4.4.1 演进历史

```
旧版 Chain（v0.1 及之前）           新版 LCEL（v0.2+）⭐推荐
─────────────────────────         ──────────────────────────
LLMChain                    →    prompt | llm | parser
SequentialChain              →    chain1 | chain2
ConversationChain            →    prompt (with memory) | llm
RetrievalQA                  →    retriever | prompt | llm
```

#### 4.4.2 使用 LCEL 构建链（推荐）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义组件
prompt = ChatPromptTemplate.from_template("用{language}解释什么是{concept}")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# 用管道符 | 串联 → 这就是 LCEL
chain = prompt | model | parser

# 调用
result = chain.invoke({
    "language": "简单中文",
    "concept": "量子计算"
})
print(result)
```

#### 4.4.3 链的执行流程

```
invoke({"language": "简单中文", "concept": "量子计算"})
  │
  ▼
┌──────────────────┐
│  PromptTemplate  │  输入: dict → 输出: PromptValue
│  "用简单中文解释  │
│  什么是量子计算"  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    ChatOpenAI    │  输入: PromptValue → 输出: AIMessage
│  调用GPT-4o API  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ StrOutputParser  │  输入: AIMessage → 输出: str
│  提取文本内容     │
└────────┬─────────┘
         │
         ▼
    "量子计算是..."  (str)
```

#### 4.4.4 复杂链示例：多步骤处理

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 第一步：生成大纲
outline_prompt = ChatPromptTemplate.from_template(
    "请为主题'{topic}'生成一个文章大纲，包含3个要点"
)

# 第二步：根据大纲写文章
article_prompt = ChatPromptTemplate.from_template(
    "根据以下大纲写一篇文章：\n{outline}\n\n要求：每个要点写2-3句话"
)

# 串联
chain = (
    outline_prompt
    | model
    | StrOutputParser()
    | (lambda outline: {"outline": outline})  # 转换格式
    | article_prompt
    | model
    | StrOutputParser()
)

result = chain.invoke({"topic": "人工智能的未来"})
```

#### 4.4.5 并行链（RunnableParallel）

```python
from langchain_core.runnables import RunnableParallel

# 同时执行多个链
parallel_chain = RunnableParallel(
    poem=ChatPromptTemplate.from_template("写一首关于{topic}的诗") | model | StrOutputParser(),
    joke=ChatPromptTemplate.from_template("讲一个关于{topic}的笑话") | model | StrOutputParser(),
    haiku=ChatPromptTemplate.from_template("写一首关于{topic}的俳句") | model | StrOutputParser(),
)

results = parallel_chain.invoke({"topic": "春天"})
print(results["poem"])
print(results["joke"])
print(results["haiku"])
```

---

### 4.5 Memory（记忆）

让对话模型"记住"之前的对话历史。

#### 4.5.1 记忆类型对比

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Types                             │
├──────────────────────┬──────────────────────────────────────┤
│ ConversationBuffer   │ 完整保留所有对话历史                   │
│ Memory               │ 简单但token消耗大                     │
├──────────────────────┼──────────────────────────────────────┤
│ ConversationBuffer   │ 只保留最近K轮对话                     │
│ WindowMemory         │ 控制token消耗                         │
├──────────────────────┼──────────────────────────────────────┤
│ ConversationSummary  │ 用LLM总结历史对话                     │
│ Memory               │ 长对话友好，但有信息损失               │
├──────────────────────┼──────────────────────────────────────┤
│ ConversationToken    │ 按token数限制历史长度                  │
│ BufferMemory         │ 精确控制token预算                     │
├──────────────────────┼──────────────────────────────────────┤
│ ConversationSummary  │ 结合摘要和最近消息                    │
│ BufferMemory         │ 平衡效果最好                          │
├──────────────────────┼──────────────────────────────────────┤
│ VectorStoreRetriever │ 从向量数据库中检索相关历史             │
│ Memory               │ 适合超长期记忆                        │
└──────────────────────┴──────────────────────────────────────┘
```

#### 4.5.2 基础用法（使用 LCEL + 手动管理历史）

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

model = ChatOpenAI(model="gpt-4o")
chain = prompt | model | StrOutputParser()

# 手动管理历史
chat_history = []

def chat(user_input: str) -> str:
    response = chain.invoke({
        "history": chat_history,
        "input": user_input,
    })
    # 更新历史
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response

# 多轮对话
print(chat("我叫小明"))        # "你好小明！..."
print(chat("我叫什么名字？"))   # "你叫小明..."
```

#### 4.5.3 使用 RunnableWithMessageHistory（推荐）

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 存储每个会话的历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 包装链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 使用 session_id 区分不同用户/会话
config = {"configurable": {"session_id": "user_123"}}

response1 = chain_with_history.invoke(
    {"input": "我是小红"},
    config=config,
)

response2 = chain_with_history.invoke(
    {"input": "我叫什么？"},
    config=config,  # 同一个session，能记住
)
```

---

### 4.6 Agents（智能体）

Agent 是 LangChain 中最强大的概念——让 LLM **自主决策** 使用哪些工具来完成任务。

#### 4.6.1 Agent 的工作原理

```
┌─────────────────────────────────────────────────────┐
│                    Agent 执行循环                     │
│                                                     │
│   用户问题 ──▶ LLM 思考 ──▶ 选择工具 ──▶ 执行工具    │
│                  ▲              │           │        │
│                  │              ▼           ▼        │
│                  └──── 观察结果 ◀── 获取结果          │
│                        │                            │
│                        ▼                            │
│                  是否足够回答？                       │
│                  ├── 否 → 继续循环                    │
│                  └── 是 → 返回最终答案                │
└─────────────────────────────────────────────────────┘
```

这就是经典的 **ReAct（Reasoning + Acting）** 模式：

```
Thought: 我需要查询今天的天气
Action: search_weather
Action Input: {"city": "北京"}
Observation: 北京今天晴，25°C
Thought: 我现在可以回答用户了
Final Answer: 北京今天天晴，气温25°C
```

#### 4.6.2 创建 Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ① 定义工具
@tool
def multiply(a: float, b: float) -> float:
    """将两个数相乘"""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """将两个数相加"""
    return a + b

@tool
def get_word_length(word: str) -> int:
    """获取一个单词的字母数量"""
    return len(word)

tools = [multiply, add, get_word_length]

# ② 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，善于使用工具解决数学问题"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent思考过程
])

# ③ 创建 Agent
model = ChatOpenAI(model="gpt-4o")
agent = create_tool_calling_agent(model, tools, prompt)

# ④ 创建 AgentExecutor（执行引擎）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # 打印思考过程
    max_iterations=10, # 最大迭代次数
)

# ⑤ 执行
result = agent_executor.invoke({
    "input": "3.14乘以7.5等于多少？再加上10呢？"
})
print(result["output"])
```

#### 4.6.3 Agent 类型

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent 类型                               │
├────────────────────────┬─────────────────────────────────────┤
│ Tool Calling Agent     │ ⭐推荐 - 使用模型原生的function      │
│ (create_tool_calling   │   calling能力，最可靠                 │
│  _agent)               │                                     │
├────────────────────────┼─────────────────────────────────────┤
│ ReAct Agent            │ 使用提示词让模型按ReAct格式推理       │
│                        │ 适用于不支持tool calling的模型        │
├────────────────────────┼─────────────────────────────────────┤
│ OpenAI Functions Agent │ (旧版) 使用OpenAI functions API      │
│                        │ 已被tool calling取代                 │
├────────────────────────┼─────────────────────────────────────┤
│ Structured Chat Agent  │ 支持多输入参数的工具                  │
├────────────────────────┼─────────────────────────────────────┤
│ Self Ask With Search   │ 分解子问题+搜索                      │
├────────────────────────┼─────────────────────────────────────┤
│ LangGraph Agent        │ ⭐新一代 - 图结构编排，最灵活         │
└────────────────────────┴─────────────────────────────────────┘
```

---

### 4.7 Tools（工具）

工具是 Agent 与外部世界交互的接口。

#### 4.7.1 自定义工具

```python
from langchain_core.tools import tool
from typing import Optional

# 方式一：使用 @tool 装饰器（最简单）
@tool
def search_weather(city: str, date: Optional[str] = None) -> str:
    """查询指定城市的天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"
        date: 日期，格式为YYYY-MM-DD，默认为今天
    """
    # 实际实现中会调用天气API
    return f"{city}{'在'+date if date else '今天'}天气晴朗，温度25°C"

print(search_weather.name)         # "search_weather"
print(search_weather.description)  # "查询指定城市的天气信息..."
print(search_weather.args_schema.schema())  # JSON Schema
```

```python
# 方式二：使用 StructuredTool（更多控制）
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=5, description="最大返回数量")

def search_func(query: str, max_results: int = 5) -> str:
    return f"搜索'{query}'，找到{max_results}条结果"

search_tool = StructuredTool.from_function(
    func=search_func,
    name="web_search",
    description="搜索互联网获取信息",
    args_schema=SearchInput,
)
```

```python
# 方式三：继承 BaseTool（完全控制）
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from langchain_core.callbacks import CallbackManagerForToolRun

class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '2+3*4'")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "计算数学表达式的结果"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            result = eval(expression)  # 注意：生产环境不要用eval
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"
```

#### 4.7.2 内置工具集成

```python
# LangChain 社区提供大量预建工具

# 搜索工具
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Python REPL
from langchain_community.tools import PythonREPLTool
python_repl = PythonREPLTool()

# 请求URL
from langchain_community.tools import RequestsGetTool
```

#### 4.7.3 工具绑定到模型

```python
# 直接将工具绑定到模型（不需要Agent也可以用）
model_with_tools = chat_model.bind_tools([multiply, add, search_weather])

response = model_with_tools.invoke("3乘以7等于多少？")

# 模型会返回工具调用信息
print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 3, 'b': 7}, 'id': 'call_xxx'}]
```

---

### 4.8 Retrieval / RAG

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是 LangChain 最重要的应用场景之一。

#### 4.8.1 RAG 完整流程

```
┌─────────── 索引阶段 (Indexing) ──────────┐
│                                          │
│  文档 → 加载 → 分割 → 嵌入 → 存入向量库   │
│                                          │
│  PDF ──┐                                 │
│  Web ──┤  Document  → TextSplitter       │
│  DB  ──┤  Loader       → Embeddings      │
│  TXT ──┘                  → VectorStore   │
│                                          │
└──────────────────────────────────────────┘

┌─────────── 检索阶段 (Retrieval) ─────────┐
│                                          │
│  用户问题 → 嵌入 → 向量相似搜索 → 相关文档 │
│                                          │
└──────────────────────────────────────────┘

┌─────────── 生成阶段 (Generation) ─────────┐
│                                          │
│  (相关文档 + 用户问题) → LLM → 回答       │
│                                          │
└──────────────────────────────────────────┘
```

#### 4.8.2 Document Loaders（文档加载器）

```python
# LangChain 支持 100+ 种文档加载器

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 网页
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()

# Word文档
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("document.docx")
docs = loader.load()

# 目录下所有文件
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.txt")
docs = loader.load()

# Notion
from langchain_community.document_loaders import NotionDBLoader

# 每个文档是一个 Document 对象
print(docs[0].page_content)  # 文本内容
print(docs[0].metadata)      # 元数据 {"source": "...", "page": 0}
```

#### 4.8.3 Text Splitters（文本分割器）

```python
# 为什么要分割？
# → LLM有上下文窗口限制
# → 小块文本检索更精准
# → 嵌入向量对短文本效果更好

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # ⭐最常用
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    HTMLSectionSplitter,
    PythonCodeTextSplitter,
)

# 递归字符分割器（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 每块最大字符数
    chunk_overlap=200,     # 块之间的重叠字符数（保持上下文连贯）
    length_function=len,   # 长度计算函数
    separators=[           # 按优先级尝试的分隔符
        "\n\n",   # 段落
        "\n",     # 换行
        "。",     # 句号
        "，",     # 逗号
        " ",      # 空格
        "",       # 字符
    ],
)

chunks = splitter.split_documents(docs)
print(f"原始文档数: {len(docs)}, 分割后: {len(chunks)}")
```

#### 4.8.4 Embeddings（嵌入模型）

```python
# 将文本转为向量（高维空间中的数值表示）

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 单个文本嵌入
vector = embeddings.embed_query("什么是机器学习？")
print(len(vector))  # 1536 维

# 批量嵌入
vectors = embeddings.embed_documents([
    "机器学习是AI的子领域",
    "深度学习使用神经网络",
])

# 其他嵌入模型
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

#### 4.8.5 Vector Stores（向量数据库）

```python
# 常见向量数据库

# ① Chroma（轻量级，适合开发测试）
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化路径
)

# ② FAISS（Meta开源，性能好）
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
# 加载
vectorstore = FAISS.load_local("faiss_index", embeddings)

# ③ Pinecone（云服务，生产级）
from langchain_pinecone import PineconeVectorStore

# ④ Milvus / Weaviate / Qdrant 等

# 相似性搜索
results = vectorstore.similarity_search("什么是深度学习？", k=3)
for doc in results:
    print(doc.page_content[:100])

# 带分数的搜索
results = vectorstore.similarity_search_with_score("深度学习", k=3)
for doc, score in results:
    print(f"相关度: {score:.4f} | {doc.page_content[:50]}")
```

#### 4.8.6 Retrievers（检索器）

```python
# Retriever 是检索的统一抽象接口

# 从向量数据库创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",       # 搜索类型
    search_kwargs={"k": 4},         # 返回数量
)

# 或使用 MMR（最大边际相关性）减少冗余
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)

# 调用
docs = retriever.invoke("什么是Transformer？")

# 多查询检索器（自动生成多个查询变体）
from langchain.retrievers import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=chat_model,
)

# 上下文压缩检索器（过滤无关内容）
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(chat_model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

#### 4.8.7 完整 RAG 链 ⭐

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# 1. 准备向量数据库（假设已有）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. RAG提示词模板
rag_prompt = ChatPromptTemplate.from_template("""
基于以下参考资料回答问题。如果资料中没有相关信息，请诚实说不知道。

参考资料：
{context}

问题：{question}

回答：
""")

# 3. 格式化检索到的文档
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# 4. 构建 RAG 链
model = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {
        "context": retriever | format_docs,      # 检索 + 格式化
        "question": RunnablePassthrough(),        # 直接传递用户问题
    }
    | rag_prompt      # 注入到提示词
    | model           # 调用LLM
    | StrOutputParser()  # 提取文本
)

# 5. 使用
answer = rag_chain.invoke("LangChain的核心组件有哪些？")
print(answer)
```

#### 4.8.8 RAG 进阶技巧

```
┌─────────────────── RAG 优化策略 ───────────────────┐
│                                                    │
│  检索优化：                                         │
│  ├── 混合搜索 (向量 + 关键词BM25)                    │
│  ├── 重排序 (Reranking) - 使用Cross-Encoder          │
│  ├── 查询改写 (Query Rewriting)                      │
│  ├── HyDE (假设性文档嵌入)                           │
│  └── 父文档检索器 (ParentDocumentRetriever)          │
│                                                    │
│  分割优化：                                         │
│  ├── 语义分割 (基于嵌入相似度)                        │
│  ├── 合理设置 chunk_size 和 overlap                  │
│  └── 保留元数据 (来源、页码等)                        │
│                                                    │
│  生成优化：                                         │
│  ├── 引用来源 (标注参考文档)                          │
│  ├── 思维链 (Chain of Thought)                      │
│  └── 自我反思 (Self-Reflection)                     │
└────────────────────────────────────────────────────┘
```

---

### 4.9 Callbacks（回调）

用于监控、日志记录、流式输出等。

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🚀 LLM 开始调用...")

    def on_llm_end(self, response, **kwargs):
        print(f"✅ LLM 调用完成")
        print(f"   Token 使用: {response.llm_output}")

    def on_llm_error(self, error, **kwargs):
        print(f"❌ LLM 调用出错: {error}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 工具调用: {serialized['name']}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"⛓️ 链开始: {serialized.get('name', 'unknown')}")

# 使用回调
model = ChatOpenAI(model="gpt-4o", callbacks=[MyHandler()])
response = model.invoke("你好")

# 或在调用时传入
response = model.invoke("你好", config={"callbacks": [MyHandler()]})
```

---

## 五、LCEL — LangChain 表达式语言

**LCEL（LangChain Expression Language）** 是 LangChain v0.2+ 的核心编程范式。

### 5.1 核心接口：Runnable

```python
# 所有 LCEL 组件都实现 Runnable 接口

class Runnable:
    def invoke(self, input)          # 同步调用
    def ainvoke(self, input)         # 异步调用
    def stream(self, input)          # 流式输出
    def astream(self, input)         # 异步流式
    def batch(self, inputs)          # 批量调用
    def abatch(self, inputs)         # 异步批量
    def astream_log(self, input)     # 流式 + 中间步骤日志
    def astream_events(self, input)  # 流式事件
```

### 5.2 管道操作符 `|`

```python
# | 操作符就是 pipe，将一个 Runnable 的输出传递给下一个的输入
chain = component_a | component_b | component_c

# 等价于
chain = component_a.pipe(component_b).pipe(component_c)

# 执行时
result = chain.invoke(input)
# 相当于
# step1 = component_a.invoke(input)
# step2 = component_b.invoke(step1)
# result = component_c.invoke(step2)
```

### 5.3 核心 Runnable 组件

```python
from langchain_core.runnables import (
    RunnablePassthrough,    # 直接传递输入（不做任何处理）
    RunnableParallel,       # 并行执行多个 Runnable
    RunnableLambda,         # 将普通函数包装为 Runnable
    RunnableBranch,         # 条件分支
    RunnableSequence,       # 顺序执行（| 的底层实现）
)
```

#### RunnablePassthrough — 透传

```python
from langchain_core.runnables import RunnablePassthrough

# 直接传递
chain = RunnablePassthrough() | some_function
# 输入什么就传什么给 some_function

# 带赋值的透传（assign）
chain = RunnablePassthrough.assign(
    extra_field=lambda x: x["name"].upper()
)
# 输入: {"name": "alice"} → 输出: {"name": "alice", "extra_field": "ALICE"}
```

#### RunnableParallel — 并行

```python
from langchain_core.runnables import RunnableParallel

# 方式一
parallel = RunnableParallel(
    upper=lambda x: x.upper(),
    lower=lambda x: x.lower(),
    length=lambda x: len(x),
)
result = parallel.invoke("Hello")
# {"upper": "HELLO", "lower": "hello", "length": 5}

# 方式二：直接用字典
parallel = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | prompt | model
```

#### RunnableLambda — 自定义函数

```python
from langchain_core.runnables import RunnableLambda

# 将任意函数包装为 Runnable
def add_exclamation(text: str) -> str:
    return text + "!"

runnable = RunnableLambda(add_exclamation)
result = runnable.invoke("Hello")  # "Hello!"

# 支持异步
import asyncio

async def async_process(text: str) -> str:
    await asyncio.sleep(1)
    return text.upper()

runnable = RunnableLambda(func=add_exclamation, afunc=async_process)
```

#### RunnableBranch — 条件分支

```python
from langchain_core.runnables import RunnableBranch

# 根据条件选择不同的处理路径
branch = RunnableBranch(
    # (条件函数, 对应的Runnable)
    (lambda x: "数学" in x["topic"], math_chain),
    (lambda x: "物理" in x["topic"], physics_chain),
    (lambda x: "化学" in x["topic"], chemistry_chain),
    general_chain,  # 默认路径（最后一个参数）
)

result = branch.invoke({"topic": "数学", "question": "1+1=?"})
```

### 5.4 LCEL 的优势

```python
chain = prompt | model | parser

# ✅ 自动获得以下能力，无需额外代码：

# 1. 流式
for chunk in chain.stream({"input": "你好"}):
    print(chunk, end="")

# 2. 异步
result = await chain.ainvoke({"input": "你好"})

# 3. 批量（自动并发）
results = chain.batch([
    {"input": "问题1"},
    {"input": "问题2"},
    {"input": "问题3"},
], config={"max_concurrency": 5})

# 4. 重试和容错
chain = chain.with_retry(stop_after_attempt=3)

# 5. 回退
fallback_chain = model_expensive | model_cheap
chain = prompt | model.with_fallbacks([cheap_model]) | parser

# 6. 流式事件（可视化每一步）
async for event in chain.astream_events({"input": "你好"}, version="v2"):
    print(event["event"], event.get("data", {}).get("chunk", ""))
```

### 5.5 完整 LCEL 示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

model = ChatOpenAI(model="gpt-4o")

# 多步骤处理链
chain = (
    # 第1步：并行准备数据
    RunnableParallel(
        topic=RunnablePassthrough(),
        context=RunnableLambda(lambda x: f"当前日期: 2025年"),
    )
    # 第2步：构建提示词
    | ChatPromptTemplate.from_template(
        "背景信息: {context}\n\n请详细解释{topic}这个话题"
    )
    # 第3步：调用模型
    | model
    # 第4步：解析输出
    | StrOutputParser()
    # 第5步：后处理
    | RunnableLambda(lambda text: {"content": text, "length": len(text)})
)

result = chain.invoke("量子计算的最新进展")
print(result["content"])
print(f"回答长度: {result['length']}")
```

---

## 六、实战案例

### 6.1 案例一：智能客服机器人

```python
"""
功能：
- 基于公司知识库回答问题 (RAG)
- 记住对话历史
- 无法回答时诚实说明
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_chroma import Chroma

# 初始化组件
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./company_kb", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 系统提示词
system_prompt = """你是公司的智能客服助手。请根据以下参考资料回答用户问题。

规则：
1. 只根据参考资料回答，不要编造信息
2. 如果资料中没有相关内容，请说"这个问题我暂时无法回答，建议联系人工客服"
3. 回答要简洁明了
4. 使用友好的语气

参考资料：
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}" for d in docs)

# 构建链
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | prompt
    | model
    | StrOutputParser()
)

# 添加历史记忆
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 使用
config = {"configurable": {"session_id": "customer_001"}}

response = chain_with_history.invoke(
    {"input": "你们的退货政策是什么？"},
    config=config,
)
print(response)
```

### 6.2 案例二：数据分析 Agent

```python
"""
功能：
- 能执行Python代码分析数据
- 能搜索互联网获取信息
- 自主决定使用哪个工具
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import subprocess

@tool
def python_executor(code: str) -> str:
    """执行Python代码并返回结果。用于数据分析、计算、生成图表等。

    Args:
        code: 要执行的Python代码
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += f"\n错误: {result.stderr}"
        return output or "代码执行完成（无输出）"
    except subprocess.TimeoutExpired:
        return "执行超时（超过30秒）"
    except Exception as e:
        return f"执行失败: {str(e)}"

@tool
def read_csv_info(filepath: str) -> str:
    """读取CSV文件的基本信息，包括列名、数据类型、行数等。

    Args:
        filepath: CSV文件路径
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    info = f"行数: {len(df)}\n列数: {len(df.columns)}\n"
    info += f"列名: {list(df.columns)}\n"
    info += f"数据类型:\n{df.dtypes.to_string()}\n"
    info += f"前5行:\n{df.head().to_string()}"
    return info

tools = [python_executor, read_csv_info]

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的数据分析师。你可以：
1. 使用Python执行代码进行数据分析
2. 读取CSV文件了解数据结构
请一步步分析，给出清晰的结论。"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({
    "input": "请分析 sales_data.csv 文件，找出销售额最高的前5个产品"
})
print(result["output"])
```

### 6.3 案例三：多模态文档问答

```python
"""
从PDF加载文档 → 分割 → 存入向量库 → 问答
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ===== 1. 索引构建（通常只需执行一次）=====
def build_index(pdf_path: str, persist_dir: str = "./chroma_db"):
    # 加载
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"加载了 {len(docs)} 页")

    # 分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"分割为 {len(chunks)} 个块")

    # 嵌入 + 存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print("索引构建完成")
    return vectorstore

# ===== 2. 问答链 =====
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )

    prompt = ChatPromptTemplate.from_template("""
你是一个文档问答助手。请严格根据提供的文档内容回答问题。

如果文档中没有相关信息，请明确告知用户。
请在回答末尾标注信息来源（页码）。

文档内容：
{context}

用户问题：{question}

详细回答：""")

    model = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "未知")
            formatted.append(f"[来源: {source}, 第{page}页]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# ===== 使用 =====
# vectorstore = build_index("my_document.pdf")
# qa_chain = create_qa_chain(vectorstore)
# answer = qa_chain.invoke("这份文档的主要结论是什么？")
# print(answer)
```

---

## 七、LangChain 生态系统

### 7.1 LangGraph — 复杂 Agent 编排

```python
"""
LangGraph 用于构建有状态、多步骤、可循环的 Agent 工作流
核心概念：图（Graph） = 节点（Node） + 边（Edge）
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

# 定义节点函数
def researcher(state: AgentState) -> AgentState:
    # 研究节点的逻辑
    return {"messages": [("researcher", "我找到了相关资料...")]}

def writer(state: AgentState) -> AgentState:
    # 写作节点的逻辑
    return {"messages": [("writer", "我已经写好了文章...")]}

def reviewer(state: AgentState) -> AgentState:
    # 审核节点的逻辑
    return {"messages": [("reviewer", "文章质量合格")]}

# 定义路由
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if "合格" in last_message[1]:
        return "end"
    return "revise"

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("research", researcher)
workflow.add_node("write", writer)
workflow.add_node("review", reviewer)

# 添加边
workflow.set_entry_point("research")
workflow.add_edge("research", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges(
    "review",
    should_continue,
    {"end": END, "revise": "write"},  # 不合格则回到写作
)

# 编译
app = workflow.compile()

# 执行
result = app.invoke({"messages": [], "next_step": ""})
```

```
LangGraph 工作流可视化：

  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Research │────▶│  Write   │────▶│  Review  │
  └──────────┘     └──────────┘     └────┬─────┘
                        ▲                │
                        │            合格？
                        │           ┌────┴────┐
                        │           │         │
                        └───── 否 ──┘    是 ──▼
                                          END
```

### 7.2 LangServe — 部署为 API

```python
# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建链
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
model = ChatOpenAI(model="gpt-4o")
chain = prompt | model

# 创建 FastAPI 应用
app = FastAPI(title="LangChain Server")

# 一行代码添加路由
add_routes(app, chain, path="/joke")

# 运行: uvicorn server:app --reload
# 自动获得:
#   POST /joke/invoke       - 同步调用
#   POST /joke/batch        - 批量调用
#   POST /joke/stream       - 流式调用
#   GET  /joke/playground   - 可视化测试界面
#   GET  /joke/input_schema - 输入Schema
#   GET  /joke/output_schema- 输出Schema
```

### 7.3 LangSmith — 监控与评估

```python
# LangSmith 提供：
# ✅ 调用链路追踪（每一步的输入/输出/耗时/Token用量）
# ✅ 调试和错误分析
# ✅ 数据集管理和自动化评估
# ✅ Prompt版本管理
# ✅ 反馈收集

# 配置（设置环境变量即可自动启用）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 之后所有 LangChain 调用都会自动上报到 LangSmith
# 在 https://smith.langchain.com 查看追踪数据
```

### 7.4 生态全景图

```
                        LangChain 生态系统
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │langchain│          │langgraph│          │langsmith│
   │  -core  │          │         │          │         │
   │         │          │ 多Agent │          │ 监控    │
   │ 核心抽象 │          │ 状态图  │          │ 调试    │
   │ LCEL    │          │ 工作流  │          │ 评估    │
   └────┬────┘          └─────────┘          └─────────┘
        │
   ┌────▼────────────────────────────────┐
   │           langchain                  │
   │  Chains / Agents / Memory / ...     │
   └────┬────────────────────────────────┘
        │
   ┌────▼────────────────────────────────┐
   │      langchain-community             │
   │  100+ 第三方集成                      │
   ├──────────────────────────────────────┤
   │  langchain-openai                    │
   │  langchain-anthropic                 │
   │  langchain-google-genai              │
   │  langchain-...                       │
   └──────────────────────────────────────┘
        │
   ┌────▼────┐
   │langserve│
   │ API部署  │
   └─────────┘
```

---

## 八、最佳实践与常见陷阱

### 8.1 ✅ 最佳实践

```
1. 使用 LCEL 而非旧版 Chain
   ─────────────────────────
   ❌ LLMChain(llm=llm, prompt=prompt)
   ✅ prompt | model | parser

2. 模型选择策略
   ─────────────
   • 简单任务 → gpt-4o-mini / claude-3-haiku（便宜快速）
   • 复杂推理 → gpt-4o / claude-3.5-sonnet（能力强）
   • 代码生成 → claude-3.5-sonnet（特别擅长）
   • 创意写作 → 适当提高 temperature

3. Prompt 工程
   ────────────
   • 使用 SystemMessage 明确角色和规则
   • 提供 Few-shot 示例
   • 用 structured_output 代替手动解析
   • 在 Prompt 中限制输出范围

4. RAG 优化
   ─────────
   • chunk_size 建议 500-1500 字符
   • chunk_overlap 建议 chunk_size 的 10-20%
   • 使用 MMR 搜索减少重复
   • 添加 Reranker 提升精度
   • 保留元数据便于引用来源

5. 错误处理
   ─────────
   • 使用 with_retry() 处理 API 暂时失败
   • 使用 with_fallbacks() 设置备用模型
   • 设置合理的 timeout 和 max_iterations

6. 成本控制
   ─────────
   • 使用 LangSmith 监控 token 用量
   • 缓存重复请求 (SQLiteCache / RedisCache)
   • 使用小模型处理简单任务
   • 控制 max_tokens 避免过长输出
```

### 8.2 ❌ 常见陷阱

```python
# 陷阱1: 忘记安装集成包
# ❌ from langchain.chat_models import ChatOpenAI  (旧路径)
# ✅ pip install langchain-openai
# ✅ from langchain_openai import ChatOpenAI

# 陷阱2: API Key 硬编码
# ❌ ChatOpenAI(api_key="sk-xxx...")
# ✅ 使用环境变量 OPENAI_API_KEY

# 陷阱3: Agent 陷入死循环
# ✅ 设置 max_iterations
agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=10,           # 限制最大迭代
    max_execution_time=60,       # 限制最大执行时间
    handle_parsing_errors=True,  # 解析错误时优雅处理
)

# 陷阱4: RAG 中 chunk 太大或太小
# 太大 → 检索不精准，包含无关信息
# 太小 → 丢失上下文，答案不完整
# ✅ 通过实验找到最佳大小

# 陷阱5: 不处理模型输出不稳定性
# LLM 输出是概率性的，可能格式不对
# ✅ 使用 with_structured_output
# ✅ 使用 PydanticOutputParser + 重试

# 陷阱6: 同步代码中调用异步方法
# ❌ await chain.ainvoke(...)  # 在非async环境
# ✅ chain.invoke(...)         # 同步环境用 invoke
# ✅ 在 async def 中才用 ainvoke
```

### 8.3 性能优化

```python
# 1. 缓存 LLM 调用
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# 2. 批量处理
results = chain.batch(
    [{"input": q} for q in questions],
    config={"max_concurrency": 10},  # 并发数
)

# 3. 异步处理
import asyncio

async def process_many(inputs):
    tasks = [chain.ainvoke(inp) for inp in inputs]
    return await asyncio.gather(*tasks)

# 4. 流式输出（提升用户体验）
for chunk in chain.stream({"input": "长问题"}):
    print(chunk, end="", flush=True)
```

---

## 九、总结与学习路线

### 9.1 核心知识图谱

```
                    LangChain 知识图谱
                          │
          ┌───────────────┼───────────────┐
          │               │               │
      基础层           核心层          应用层
          │               │               │
    ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
    │ Model I/O │   │   LCEL    │   │    RAG    │
    │ Prompts   │   │  Chains   │   │  Agents   │
    │ Parsers   │   │  Memory   │   │  LangGraph│
    └───────────┘   │ Callbacks │   │ LangServe │
                    └───────────┘   └───────────┘
```

### 9.2 推荐学习路线

```
Week 1: 基础入门
├── 安装 LangChain + 配置 API Key
├── 学习 Model I/O（ChatModel 调用）
├── 掌握 PromptTemplate
└── 理解 OutputParser

Week 2: 核心概念
├── 深入理解 LCEL（Runnable 接口）
├── 用 | 管道符构建链
├── 学习 RunnableParallel / PassThrough
└── 掌握 Memory 机制

Week 3: RAG 实战
├── Document Loader + Text Splitter
├── Embeddings + Vector Store
├── 构建完整 RAG 管线
└── 优化检索质量

Week 4: Agent & 进阶
├── 自定义 Tool
├── 构建 Agent（Tool Calling）
├── 学习 LangGraph（复杂工作流）
└── 使用 LangSmith 监控调试

Week 5+: 生产实践
├── LangServe 部署
├── 错误处理和容错
├── 性能优化
└── 构建完整项目
```

### 9.3 参考资源

| 资源           | 链接                                               | 说明         |
| -------------- | -------------------------------------------------- | ------------ |
| 官方文档       | https://python.langchain.com/docs                  | 最权威的参考 |
| API参考        | https://api.python.langchain.com                   | 详细API文档  |
| GitHub         | https://github.com/langchain-ai/langchain          | 源码         |
| LangSmith      | https://smith.langchain.com                        | 监控平台     |
| LangGraph 文档 | https://langchain-ai.github.io/langgraph/          | 图工作流     |
| Cookbook       | https://github.com/langchain-ai/langchain/cookbook | 示例代码     |

### 9.4 安装速查

```bash
# 核心包
pip install langchain langchain-core

# 常用集成
pip install langchain-openai          # OpenAI
pip install langchain-anthropic       # Anthropic Claude
pip install langchain-google-genai    # Google Gemini
pip install langchain-community       # 社区集成

# RAG相关
pip install langchain-chroma          # Chroma向量库
pip install langchain-text-splitters  # 文本分割
pip install pypdf                     # PDF加载

# Agent / 工具
pip install duckduckgo-search         # DuckDuckGo搜索
pip install wikipedia                 # Wikipedia

# 部署
pip install langserve                 # API部署
pip install langgraph                 # 图工作流

# 开发工具
pip install langsmith                 # 监控调试
```

---

> **最后的建议**：LangChain 更新迭代非常快，建议以 **官方文档** 为准。核心思想不变：**组件化 + LCEL编排 + 工具集成**。先理解基础概念，再按需深入特定模块。动手实践是最好的学习方式。

---
