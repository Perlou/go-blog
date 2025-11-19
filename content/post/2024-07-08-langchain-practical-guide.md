+++
date = '2024-07-08T09:15:00+08:00'
draft = false
title = '使用LangChain构建AI应用的实战经验'
categories = ['AI测试']
tags = ['AI测试']
+++

LangChain 是一个强大的框架，用于开发由大语言模型驱动的应用。本文分享使用 LangChain 构建 AI 应用的实战经验。

## LangChain 简介

LangChain 提供了一系列组件和工具，简化了 AI 应用的开发流程：

- **Models**：与各种 LLM 交互
- **Prompts**：提示模板管理
- **Chains**：组合多个操作
- **Agents**：智能代理
- **Memory**：对话记忆管理

## 基础用法

### 1. 安装配置

```bash
pip install langchain openai chromadb
```

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# 初始化模型
llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(model="gpt-3.5-turbo")
```

### 2. 提示模板

使用模板提高提示的复用性：

```python
from langchain.prompts import PromptTemplate

template = """
你是一位{role}。
请根据以下信息回答问题：

背景：{context}
问题：{question}

回答：
"""

prompt = PromptTemplate(
    input_variables=["role", "context", "question"],
    template=template
)

formatted_prompt = prompt.format(
    role="Python专家",
    context="用户想学习异步编程",
    question="什么是asyncio？"
)
```

### 3. 链式调用

组合多个步骤：

```python
from langchain.chains import LLMChain, SequentialChain

# 第一步：生成大纲
outline_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="为'{topic}'生成文章大纲"
    ),
    output_key="outline"
)

# 第二步：根据大纲写文章
article_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["outline"],
        template="根据以下大纲写一篇文章：\n{outline}"
    ),
    output_key="article"
)

# 组合链
overall_chain = SequentialChain(
    chains=[outline_chain, article_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"]
)

result = overall_chain({"topic": "机器学习入门"})
```

## 高级功能

### 1. 智能代理

创建能使用工具的 AI 代理：

```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# 定义工具
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="搜索最新信息"
    ),
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="数学计算"
    )
]

# 初始化代理
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 使用代理
result = agent.run("2024年世界人口是多少？")
```

### 2. 记忆管理

实现上下文记忆：

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 多轮对话
conversation.predict(input="我的名字是小明")
conversation.predict(input="我刚才说我叫什么？")
# 输出：你说你叫小明
```

### 3. 向量数据库

实现语义搜索：

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 准备文档
documents = [
    "Python是一种高级编程语言",
    "JavaScript用于Web开发",
    "Go语言适合并发编程"
]

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
docs = text_splitter.create_documents(documents)

# 创建向量数据库
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# 相似度搜索
results = db.similarity_search("Web开发用什么语言？")
print(results[0].page_content)  # JavaScript用于Web开发
```

### 4. RAG（检索增强生成）

```python
from langchain.chains import RetrievalQA

# 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# 基于文档回答问题
answer = qa_chain.run("哪种语言适合并发编程？")
print(answer)  # Go语言适合并发编程
```

## 实战案例：文档问答系统

```python
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

class DocumentQA:
    def __init__(self, doc_path):
        # 加载文档
        loader = TextLoader(doc_path)
        documents = loader.load()

        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # 创建向量数据库
        embeddings = OpenAIEmbeddings()
        self.db = Chroma.from_documents(texts, embeddings)

        # 创建QA链
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3})
        )

    def ask(self, question):
        return self.qa.run(question)

# 使用
doc_qa = DocumentQA("technical_doc.txt")
answer = doc_qa.ask("API的认证方式是什么？")
```

## 性能优化

### 1. 缓存

```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

### 2. 批量处理

```python
questions = ["问题1", "问题2", "问题3"]
results = llm.generate(questions)
```

### 3. 流式输出

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

## 最佳实践

1. **合理设置 temperature**：创意任务用高值，事实性任务用低值
2. **优化提示词**：清晰、具体、包含示例
3. **错误处理**：处理 API 限流、超时等异常
4. **成本控制**：监控 token 使用，使用缓存
5. **安全考虑**：验证用户输入，防止注入攻击

## 总结

LangChain 大大简化了 AI 应用的开发。通过合理使用其组件和工具，可以快速构建强大的 AI 应用。关键是理解各个组件的作用，并根据实际需求灵活组合。
