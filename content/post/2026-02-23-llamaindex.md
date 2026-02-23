+++
date = '2026-02-23T17:23:25+08:00'
draft = false
title = 'LlamaIndex 深入解析：从零到精通'
image = '/images/bg/glacier-river.jpg'
categories = ['AI', 'Agent']
tags = ['AI', 'LlamaIndex']
+++

# LlamaIndex 深入解析：从零到精通

---

## 目录

- [一、LlamaIndex 概述](#一llamaindex-概述)
- [二、核心架构与设计理念](#二核心架构与设计理念)
- [三、安装与环境配置](#三安装与环境配置)
- [四、核心概念详解](#四核心概念详解)
- [五、数据加载（Data Connectors）](#五数据加载data-connectors)
- [六、数据索引（Indexes）](#六数据索引indexes)
- [七、存储机制（Storage）](#七存储机制storage)
- [八、查询引擎（Query Engine）](#八查询引擎query-engine)
- [九、聊天引擎（Chat Engine）](#九聊天引擎chat-engine)
- [十、Agent 智能体](#十agent-智能体)
- [十一、高级特性](#十一高级特性)
- [十二、完整实战案例](#十二完整实战案例)
- [十三、性能优化与最佳实践](#十三性能优化与最佳实践)
- [十四、与 LangChain 的对比](#十四与-langchain-的对比)
- [十五、常见问题与排错](#十五常见问题与排错)
- [十六、总结与学习路线](#十六总结与学习路线)

---

## 一、LlamaIndex 概述

### 1.1 什么是 LlamaIndex？

**LlamaIndex**（前身为 GPT Index）是一个用于构建 **LLM（大语言模型）应用** 的数据框架。它的核心使命是：

> **将你的私有数据与大语言模型连接起来，实现基于自有数据的智能问答、检索与推理。**

```
┌─────────────────────────────────────────────────────┐
│                  LlamaIndex 定位                      │
│                                                       │
│   私有数据（PDF/数据库/API/网页...）                    │
│         │                                             │
│         ▼                                             │
│   ┌───────────┐    ┌───────────┐    ┌──────────────┐ │
│   │ 数据加载   │───▶│ 索引构建   │───▶│ 查询/对话     │ │
│   │ Ingestion │    │ Indexing  │    │ Querying     │ │
│   └───────────┘    └───────────┘    └──────────────┘ │
│                                           │           │
│                                           ▼           │
│                                     LLM (GPT/Claude)  │
└─────────────────────────────────────────────────────┘
```

### 1.2 解决了什么问题？

| 痛点                 | LlamaIndex 方案               |
| -------------------- | ----------------------------- |
| LLM 没有你的私有数据 | 通过 RAG 将私有数据注入上下文 |
| LLM 上下文窗口有限   | 智能检索最相关的片段          |
| 数据源格式多样       | 统一的数据加载器抽象          |
| 构建 RAG 流程复杂    | 高层抽象，5行代码即可搭建     |
| 需要灵活定制         | 底层完全可控，支持深度定制    |

### 1.3 核心能力一览

```
LlamaIndex 五大核心能力
├── 🔌 Data Connectors     ─── 连接 160+ 数据源
├── 📊 Data Indexes        ─── 结构化数据索引
├── 🔍 Query Engines       ─── 自然语言查询接口
├── 💬 Chat Engines        ─── 多轮对话引擎
└── 🤖 Agents             ─── 自主决策智能体
```

---

## 二、核心架构与设计理念

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        LlamaIndex 架构                          │
│                                                                  │
│  ┌──────────────────── Ingestion Pipeline ────────────────────┐  │
│  │                                                             │  │
│  │  Data Sources ──▶ Reader ──▶ Transformations ──▶ Index     │  │
│  │  (PDF,DB,API)    (Loader)   (Chunk,Embed...)   (Storage)   │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────── Querying Stage ────────────────────────┐  │
│  │                                                             │  │
│  │  User Query ──▶ Retriever ──▶ Response Synthesizer ──▶ Out │  │
│  │                    │              │                         │  │
│  │                    ▼              ▼                         │  │
│  │              Vector Store      LLM API                     │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─────────────── Supporting Components ──────────────────────┐  │
│  │                                                             │  │
│  │  LLM  │  Embeddings  │  Prompt Templates  │  Callbacks     │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 两大核心阶段

#### 阶段一：数据摄取（Indexing Stage）

```python
# 概念流程
Raw Data → Documents → Nodes → Embeddings → Index（存入 VectorStore）
```

| 步骤                 | 说明                            |
| -------------------- | ------------------------------- |
| **加载 (Load)**      | 将原始数据转为 `Document` 对象  |
| **转换 (Transform)** | 分块（Chunking）、提取元数据等  |
| **嵌入 (Embed)**     | 调用 Embedding 模型将文本向量化 |
| **索引 (Index)**     | 将向量存入索引结构中            |

#### 阶段二：查询（Querying Stage）

```python
# 概念流程
User Query → Retriever（检索相关 Nodes）→ Post-Processing → Response Synthesis → Answer
```

| 步骤                      | 说明                                  |
| ------------------------- | ------------------------------------- |
| **检索 (Retrieve)**       | 根据查询从索引中获取最相关的 Nodes    |
| **后处理 (Post-process)** | 重排序、过滤、关键词高亮等            |
| **合成 (Synthesize)**     | 将检索结果和原始查询交给 LLM 生成回答 |

### 2.3 核心数据抽象

```
┌──────────────────────────────────────────────────┐
│              LlamaIndex 数据模型                   │
│                                                    │
│  Document                                          │
│  ├── text: str           # 文档原始文本             │
│  ├── metadata: dict      # 元数据                  │
│  ├── doc_id: str         # 文档唯一标识             │
│  └── relationships: dict # 与其他节点的关系          │
│                                                    │
│  Node (TextNode)                                   │
│  ├── text: str           # 分块后的文本             │
│  ├── metadata: dict      # 继承+额外元数据          │
│  ├── embedding: List     # 向量表示                 │
│  ├── node_id: str        # 节点唯一标识             │
│  └── relationships:                                │
│      ├── SOURCE          # 来源 Document            │
│      ├── PREVIOUS        # 前一个 Node              │
│      └── NEXT            # 后一个 Node              │
└──────────────────────────────────────────────────┘
```

> **Document** 是原始文档的抽象，**Node** 是 Document 分割后的最小检索单元。

---

## 三、安装与环境配置

### 3.1 安装方式

```bash
# 基础安装（核心功能）
pip install llama-index

# 完整安装（新版本模块化后）
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma

# 常用组合一键安装
pip install llama-index llama-index-readers-file llama-index-llms-openai
```

### 3.2 版本说明（v0.10+ 重大变更）

从 **v0.10** 开始，LlamaIndex 采用了 **模块化架构**：

```
llama-index（元包）
├── llama-index-core          # 核心库
├── llama-index-llms-*        # LLM 集成
├── llama-index-embeddings-*  # Embedding 模型集成
├── llama-index-vector-stores-* # 向量数据库集成
├── llama-index-readers-*     # 数据读取器
└── llama-index-callbacks-*   # 回调/可观测性
```

### 3.3 配置 LLM 和 Embedding

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 全局配置
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 1024
Settings.chunk_overlap = 200
```

#### 使用其他模型（如本地模型）

```python
# 使用 Ollama 本地模型
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
```

```python
# 使用 Azure OpenAI
from llama_index.llms.azure_openai import AzureOpenAI

Settings.llm = AzureOpenAI(
    engine="gpt-4",
    azure_endpoint="https://xxx.openai.azure.com/",
    api_key="your-key",
    api_version="2024-02-01",
)
```

---

## 四、核心概念详解

### 4.1 RAG（检索增强生成）原理

LlamaIndex 最核心的应用场景就是 **RAG**：

```
            RAG 工作流程

    ┌──────────────────────────────────────────┐
    │  离线阶段（Indexing）                      │
    │                                           │
    │  Documents ──▶ Chunk ──▶ Embed ──▶ Store  │
    │                                           │
    └──────────────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────┐
    │  在线阶段（Querying）                      │
    │                                           │
    │  Query ──▶ Embed ──▶ Search ──▶ Top-K    │
    │                                    │      │
    │           ┌────────────────────────┘      │
    │           ▼                               │
    │  [Context + Query] ──▶ LLM ──▶ Answer    │
    │                                           │
    └──────────────────────────────────────────┘
```

### 4.2 Document 与 Node

```python
from llama_index.core import Document
from llama_index.core.schema import TextNode

# 手动创建 Document
doc = Document(
    text="LlamaIndex 是一个强大的框架...",
    metadata={
        "source": "tutorial.md",
        "author": "OpenAI",
        "date": "2024-01-01"
    },
    doc_id="doc_001"
)

# 手动创建 Node
node = TextNode(
    text="LlamaIndex 是一个强大的框架",
    metadata={"source": "tutorial.md"},
    id_="node_001"
)

# Document 和 Node 的关系
# Document 会被自动分割成多个 Node
# 每个 Node 保存对 Source Document 的引用
```

### 4.3 Node Parser（文本分割器）

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
)

# 1. 句子分割器（最常用）
splitter = SentenceSplitter(
    chunk_size=1024,      # 每个块的最大 token 数
    chunk_overlap=200,    # 块之间的重叠 token 数
    separator=" ",        # 分隔符
)

# 2. Token 分割器
splitter = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
)

# 3. 语义分割器（根据语义相似度分割）
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
)

# 4. 层次分割器（多粒度）
splitter = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # 大→中→小
)

# 使用分割器
nodes = splitter.get_nodes_from_documents([doc])
```

### 4.4 Embedding 模型

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# OpenAI Embedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # 或 text-embedding-3-large
    dimensions=1536,
)

# HuggingFace 本地 Embedding
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文推荐
    cache_folder="./models",
    device="cuda",  # 使用 GPU
)

# 手动获取向量
vector = embed_model.get_text_embedding("Hello World")
print(f"向量维度: {len(vector)}")  # 1536

# 批量获取
vectors = embed_model.get_text_embedding_batch(["Hello", "World"])
```

---

## 五、数据加载（Data Connectors）

### 5.1 数据加载架构

```
                    Data Connectors 全景
┌─────────────────────────────────────────────────────┐
│                                                      │
│  📄 文件类                                            │
│  ├── SimpleDirectoryReader  (PDF/TXT/DOCX/CSV...)   │
│  ├── PDFReader                                       │
│  ├── DocxReader                                      │
│  └── CSVReader                                       │
│                                                      │
│  🌐 网络类                                            │
│  ├── SimpleWebPageReader                             │
│  ├── BeautifulSoupWebReader                          │
│  └── TrafilaturaWebReader                            │
│                                                      │
│  🗄️ 数据库类                                          │
│  ├── DatabaseReader (SQL)                            │
│  ├── MongoDBReader                                   │
│  └── ElasticsearchReader                             │
│                                                      │
│  ☁️ 云服务类                                           │
│  ├── GoogleDocsReader                                │
│  ├── SlackReader                                     │
│  ├── NotionPageReader                                │
│  └── S3Reader                                        │
│                                                      │
│  🔗 LlamaHub: 160+ 社区贡献的 Reader                  │
│     https://llamahub.ai                              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 5.2 SimpleDirectoryReader（最常用）

```python
from llama_index.core import SimpleDirectoryReader

# 基础用法 - 读取整个目录
documents = SimpleDirectoryReader(
    input_dir="./data",
).load_data()

# 进阶配置
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,                    # 递归读取子目录
    required_exts=[".pdf", ".txt", ".md"],  # 限定文件类型
    exclude=["*.tmp"],                 # 排除文件
    filename_as_id=True,               # 用文件名作为 doc_id
    num_files_limit=100,               # 限制文件数量
).load_data()

# 读取单个文件
documents = SimpleDirectoryReader(
    input_files=["./report.pdf", "./data.csv"]
).load_data()

# 查看加载结果
for doc in documents:
    print(f"Doc ID: {doc.doc_id}")
    print(f"Metadata: {doc.metadata}")
    print(f"Text (前100字): {doc.text[:100]}")
    print("---")
```

### 5.3 从不同数据源加载

```python
# ===== 1. 从网页加载 =====
from llama_index.readers.web import SimpleWebPageReader

loader = SimpleWebPageReader(html_to_text=True)
documents = loader.load_data(
    urls=["https://example.com/page1", "https://example.com/page2"]
)

# ===== 2. 从数据库加载 =====
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    uri="postgresql://user:pass@localhost/dbname"
)
documents = reader.load_data(
    query="SELECT title, content FROM articles WHERE published = true"
)

# ===== 3. 从 Notion 加载 =====
from llama_index.readers.notion import NotionPageReader

reader = NotionPageReader(integration_token="your-token")
documents = reader.load_data(page_ids=["page-id-1", "page-id-2"])

# ===== 4. 从 S3 加载 =====
from llama_index.readers.s3 import S3Reader

reader = S3Reader(
    bucket="my-bucket",
    prefix="documents/",
    aws_access_id="xxx",
    aws_access_secret="xxx",
)
documents = reader.load_data()

# ===== 5. 手动构建 Document =====
from llama_index.core import Document

documents = [
    Document(text="第一篇文档的内容...", metadata={"source": "manual"}),
    Document(text="第二篇文档的内容...", metadata={"source": "manual"}),
]
```

### 5.4 Ingestion Pipeline（数据摄取管线）

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)

# 构建完整的数据处理管线
pipeline = IngestionPipeline(
    transformations=[
        # 步骤1: 文本分割
        SentenceSplitter(chunk_size=1024, chunk_overlap=200),

        # 步骤2: 自动提取标题元数据
        TitleExtractor(nodes=5),

        # 步骤3: 自动提取摘要
        SummaryExtractor(summaries=["prev", "self"]),

        # 步骤4: 自动提取关键词
        KeywordExtractor(keywords=10),

        # 步骤5: 生成 Embedding
        OpenAIEmbedding(model="text-embedding-3-small"),
    ]
)

# 运行管线
nodes = pipeline.run(documents=documents, show_progress=True)

# 管线支持缓存（避免重复处理）
from llama_index.core.ingestion import IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore

pipeline = IngestionPipeline(
    transformations=[...],
    cache=IngestionCache(),          # 启用缓存
    docstore=SimpleDocumentStore(),  # 去重
)
```

---

## 六、数据索引（Indexes）

### 6.1 索引类型全景

```
                    LlamaIndex 索引类型
┌─────────────────────────────────────────────────────┐
│                                                      │
│  1. VectorStoreIndex （向量索引）⭐ 最常用            │
│     └── 将节点转为向量，基于语义相似度检索              │
│                                                      │
│  2. SummaryIndex（摘要索引，原 ListIndex）             │
│     └── 顺序遍历所有节点，适合摘要任务                 │
│                                                      │
│  3. TreeIndex（树状索引）                              │
│     └── 自底向上构建摘要树，适合层次化查询              │
│                                                      │
│  4. KeywordTableIndex（关键词索引）                    │
│     └── 基于关键词匹配检索                             │
│                                                      │
│  5. KnowledgeGraphIndex（知识图谱索引）                │
│     └── 构建三元组知识图谱                             │
│                                                      │
│  6. PropertyGraphIndex（属性图索引）🆕                 │
│     └── 更灵活的图结构索引                             │
│                                                      │
│  7. DocumentSummaryIndex（文档摘要索引）               │
│     └── 先摘要后检索，两阶段检索                       │
│                                                      │
│  8. ComposableGraph（组合索引）                        │
│     └── 多种索引组合使用                               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 6.2 VectorStoreIndex（核心索引）

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# ===== 最简用法（5行代码实现 RAG）=====
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("这篇文档讲了什么？")
print(response)

# ===== 详细配置 =====
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,         # 显示进度条
    # 以下参数通过 Settings 全局配置或在此指定
    # embed_model=embed_model,
    # transformations=[splitter],
)

# 从已有 Nodes 构建索引
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes, show_progress=True)

# 增量添加文档
index.insert(new_document)

# 删除文档
index.delete_ref_doc("doc_id_to_delete")

# 更新文档
index.update_ref_doc(updated_document)
```

### 6.3 使用外部向量数据库

```python
# ===== Chroma =====
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# 创建 Chroma 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

# 从已有向量库加载索引（无需重新 embed）
index = VectorStoreIndex.from_vector_store(vector_store)


# ===== Pinecone =====
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

pc = Pinecone(api_key="your-api-key")
pinecone_index = pc.Index("my-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


# ===== Qdrant =====
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
)


# ===== Milvus =====
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="my_collection",
    dim=1536,
)


# ===== Weaviate =====
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

client = weaviate.Client("http://localhost:8080")
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="MyIndex",
)
```

### 6.4 其他索引类型

```python
from llama_index.core import (
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    DocumentSummaryIndex,
)

# ===== SummaryIndex（遍历所有节点）=====
# 适合：需要全文摘要或无法预判检索目标的场景
summary_index = SummaryIndex.from_documents(documents)
query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"  # 树状摘要
)

# ===== TreeIndex（树状层次索引）=====
# 适合：层次化文档、需要不同粒度回答的场景
tree_index = TreeIndex.from_documents(documents, num_children=10)
query_engine = tree_index.as_query_engine(
    child_branch_factor=2  # 每次检索展开的分支数
)

# ===== KeywordTableIndex（关键词索引）=====
# 适合：基于关键词的精确匹配场景
keyword_index = KeywordTableIndex.from_documents(documents)
query_engine = keyword_index.as_query_engine()

# ===== KnowledgeGraphIndex（知识图谱）=====
# 适合：实体关系提取、结构化知识查询
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,  # 每个块最多提取10个三元组
    include_embeddings=True,
)

# ===== DocumentSummaryIndex（文档级摘要索引）=====
# 适合：大量文档，先定位文档再深入检索
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    response_synthesizer=get_response_synthesizer(
        response_mode="tree_summarize"
    ),
)
```

---

## 七、存储机制（Storage）

### 7.1 存储架构

```
               LlamaIndex 存储架构
┌──────────────────────────────────────────┐
│            StorageContext                  │
│  ┌────────────────────────────────────┐  │
│  │  DocStore      (文档存储)           │  │
│  │  └── 存储原始 Document/Node 数据    │  │
│  ├────────────────────────────────────┤  │
│  │  IndexStore    (索引元数据存储)      │  │
│  │  └── 存储索引结构信息               │  │
│  ├────────────────────────────────────┤  │
│  │  VectorStore   (向量存储)           │  │
│  │  └── 存储 Embedding 向量           │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### 7.2 持久化存储

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# ===== 保存索引到磁盘 =====
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 持久化到指定目录
index.storage_context.persist(persist_dir="./storage")

# ===== 从磁盘加载索引 =====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# 继续使用
query_engine = index.as_query_engine()
response = query_engine.query("你的问题")
```

### 7.3 使用外部存储后端

```python
# ===== Redis 作为 DocStore =====
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore

storage_context = StorageContext.from_defaults(
    docstore=RedisDocumentStore.from_host_and_port(
        host="localhost", port=6379
    ),
    index_store=RedisIndexStore.from_host_and_port(
        host="localhost", port=6379
    ),
)

# ===== MongoDB 作为 DocStore =====
from llama_index.storage.docstore.mongodb import MongoDocumentStore

storage_context = StorageContext.from_defaults(
    docstore=MongoDocumentStore.from_uri(
        uri="mongodb://localhost:27017",
        db_name="llama_index"
    ),
)
```

---

## 八、查询引擎（Query Engine）

### 8.1 查询引擎架构

```
                Query Engine 内部流程

User Query ─────────────────────────────────────▶ Response
     │                                                ▲
     │ ①                                              │ ④
     ▼                                                │
 ┌──────────┐   ②    ┌──────────────┐   ③   ┌───────────────┐
 │ Retriever │──────▶│ Node          │─────▶│ Response       │
 │           │       │ PostProcessors│      │ Synthesizer    │
 └──────────┘       └──────────────┘      └───────────────┘
     │                     │                       │
     ▼                     ▼                       ▼
 Vector Store        重排/过滤/高亮            LLM 生成回答
```

### 8.2 基础查询引擎

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# ===== 基础查询 =====
query_engine = index.as_query_engine()
response = query_engine.query("什么是LlamaIndex？")
print(response)

# 访问 response 的详细信息
print(response.response)          # 回答文本
print(response.source_nodes)      # 检索到的源节点
print(response.metadata)          # 元数据

# 遍历源节点
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}")
    print(f"Metadata: {node.metadata}")
    print("---")
```

### 8.3 查询引擎配置

```python
from llama_index.core.response_synthesizers import ResponseMode

query_engine = index.as_query_engine(
    # === 检索配置 ===
    similarity_top_k=5,               # 检索 Top-K 个结果

    # === 回答合成模式 ===
    response_mode=ResponseMode.COMPACT,  # 回答模式（见下表）

    # === 流式输出 ===
    streaming=True,                    # 启用流式

    # === 其他 ===
    verbose=True,                      # 详细日志
)

# 流式输出使用
response = query_engine.query("解释一下RAG")
for text in response.response_gen:
    print(text, end="", flush=True)
```

#### Response Mode 对比

| 模式                 | 说明                           | 适用场景     |
| -------------------- | ------------------------------ | ------------ |
| `REFINE`             | 逐个节点迭代优化答案           | 需要精确回答 |
| `COMPACT`            | 合并节点后一次性生成（默认）   | 通用场景     |
| `TREE_SUMMARIZE`     | 树状递归摘要                   | 长文档摘要   |
| `SIMPLE_SUMMARIZE`   | 简单截断后摘要                 | 快速摘要     |
| `NO_TEXT`            | 只返回检索到的节点，不调用 LLM | 仅检索       |
| `ACCUMULATE`         | 对每个节点独立回答后汇总       | 多角度回答   |
| `COMPACT_ACCUMULATE` | COMPACT + ACCUMULATE           | 多角度+紧凑  |

### 8.4 自定义 Retriever

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import get_response_synthesizer

# 自定义 Retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    # 过滤条件
    filters=MetadataFilters(
        filters=[
            MetadataFilter(key="source", value="report.pdf"),
            MetadataFilter(key="year", value="2024", operator=">="),
        ],
        condition=FilterCondition.AND,
    ),
)

# 自定义 Response Synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    streaming=True,
)

# 组装查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)
```

### 8.5 Node PostProcessors（后处理器）

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    MetadataReplacementPostProcessor,
    LongContextReorder,
    SentenceTransformerRerank,
)

# 1. 相似度过滤（丢弃低于阈值的结果）
similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.7)

# 2. 关键词过滤
keyword_filter = KeywordNodePostprocessor(
    required_keywords=["LlamaIndex"],
    exclude_keywords=["deprecated"],
)

# 3. 重排序（使用 Cross-Encoder 精排）
# pip install llama-index-postprocessor-cohere-rerank
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    api_key="your-cohere-key",
    top_n=3,  # 重排后取 Top-3
)

# 或使用 Sentence Transformer 重排序
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3,
)

# 4. 长上下文重排（解决 Lost in the Middle 问题）
reorder = LongContextReorder()

# 组合使用
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        similarity_filter,
        reranker,       # 先重排
        reorder,        # 再重排列顺序
    ],
)
```

### 8.6 SubQuestion Query Engine（子问题查询引擎）

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 假设我们有多个索引（不同数据源）
tool_list = [
    QueryEngineTool(
        query_engine=financial_engine,
        metadata=ToolMetadata(
            name="financial_report",
            description="包含2024年财务报告的数据，包括营收、利润等",
        ),
    ),
    QueryEngineTool(
        query_engine=product_engine,
        metadata=ToolMetadata(
            name="product_docs",
            description="包含产品文档和技术规格",
        ),
    ),
]

# 子问题查询引擎会自动将复杂问题拆解为多个子问题
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tool_list,
    verbose=True,
)

response = query_engine.query(
    "对比一下我们的产品技术指标和财务表现之间的关系"
)
# 内部会拆解为：
# 子问题1: 产品技术指标是什么？→ 查询 product_docs
# 子问题2: 财务表现如何？→ 查询 financial_report
# 最终合成: 综合两个答案回答原始问题
```

---

## 九、聊天引擎（Chat Engine）

### 9.1 Chat Engine vs Query Engine

```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│  Query Engine:  单轮问答，无记忆                           │
│  ┌──────┐    ┌──────────┐    ┌──────┐                    │
│  │Query │───▶│  Engine   │───▶│Answer│                    │
│  └──────┘    └──────────┘    └──────┘                    │
│                                                           │
│  Chat Engine:  多轮对话，有记忆                            │
│  ┌──────┐    ┌──────────┐    ┌──────┐                    │
│  │Msg 1 │───▶│          │───▶│Reply1│                    │
│  │Msg 2 │───▶│  Engine  │───▶│Reply2│    ← 记住上下文     │
│  │Msg 3 │───▶│ +Memory  │───▶│Reply3│                    │
│  └──────┘    └──────────┘    └──────┘                    │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 9.2 Chat Engine 模式

```python
from llama_index.core.chat_engine import ChatMode

# ===== 1. BEST（自动选择最佳模式）=====
chat_engine = index.as_chat_engine(chat_mode=ChatMode.BEST)

# ===== 2. CONDENSE_QUESTION（问题浓缩模式）=====
# 将对话历史 + 新问题 → 浓缩为独立问题 → 查询索引
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_QUESTION,
    verbose=True,
)

# ===== 3. CONTEXT（上下文模式）=====
# 每次都检索相关上下文，与对话历史一起发给 LLM
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT,
    system_prompt="你是一个专业的技术助手，基于提供的文档回答问题。",
)

# ===== 4. CONDENSE_PLUS_CONTEXT =====
# 先浓缩问题，再检索上下文（推荐）
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
    verbose=True,
)

# ===== 5. REACT（使用 ReAct Agent）=====
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.REACT,
    verbose=True,
)
```

### 9.3 使用 Chat Engine

```python
# 创建聊天引擎
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    verbose=True,
)

# 多轮对话
response1 = chat_engine.chat("LlamaIndex 是什么？")
print(response1)

response2 = chat_engine.chat("它有哪些核心功能？")  # 能理解"它"指代 LlamaIndex
print(response2)

response3 = chat_engine.chat("第一个功能能详细解释一下吗？")
print(response3)

# 流式对话
response = chat_engine.stream_chat("给我一个完整的示例")
for token in response.response_gen:
    print(token, end="", flush=True)

# 重置对话历史
chat_engine.reset()

# 查看对话历史
print(chat_engine.chat_history)
```

### 9.4 自定义 Chat Engine 记忆

```python
from llama_index.core.memory import ChatMemoryBuffer

# 自定义记忆缓冲区
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,  # 记忆 token 上限
)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "你是一个友好的AI助手。"
        "请根据提供的上下文回答问题。"
        "如果不确定，请诚实地说不知道。"
    ),
    verbose=True,
)
```

---

## 十、Agent 智能体

### 10.1 Agent 概述

```
                    Agent 工作原理

┌──────────────────────────────────────────────────┐
│                                                    │
│  User Input                                        │
│      │                                             │
│      ▼                                             │
│  ┌──────────┐                                      │
│  │  Agent    │ ◄── 推理循环（ReAct / Function Call）│
│  │ (LLM)    │                                      │
│  └────┬─────┘                                      │
│       │ 决定使用哪个工具                              │
│       ▼                                             │
│  ┌────────────────────────────────────┐            │
│  │         Tools (工具集)              │            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐      │            │
│  │  │Query │ │计算器 │ │API   │ ... │            │
│  │  │Engine│ │      │ │调用  │      │            │
│  │  └──────┘ └──────┘ └──────┘      │            │
│  └────────────────────────────────────┘            │
│       │                                             │
│       ▼ 获取工具结果                                 │
│  ┌──────────┐                                      │
│  │  Agent    │ ◄── 判断是否需要继续使用工具           │
│  │ (LLM)    │                                      │
│  └────┬─────┘                                      │
│       │                                             │
│       ▼                                             │
│   Final Response                                   │
│                                                    │
└──────────────────────────────────────────────────┘
```

### 10.2 ReAct Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata

# ===== 创建查询工具 =====
query_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="knowledge_base",
        description="包含公司内部知识库的文档，可以回答关于公司产品、政策等问题",
    ),
)

# ===== 创建自定义函数工具 =====
def multiply(a: float, b: float) -> float:
    """将两个数字相乘并返回结果。"""
    return a * b

def add(a: float, b: float) -> float:
    """将两个数字相加并返回结果。"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# ===== 创建 ReAct Agent =====
agent = ReActAgent.from_tools(
    tools=[query_tool, multiply_tool, add_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True,       # 打印推理过程
    max_iterations=10,  # 最大推理步数
)

# 使用 Agent
response = agent.chat("公司去年的营收是多少？如果增长20%，明年预计是多少？")
# Agent 会：
# 1. 思考：需要先查询公司营收
# 2. 行动：调用 knowledge_base 工具
# 3. 观察：获取营收数据（比如 1000万）
# 4. 思考：需要计算 1000万 * 1.2
# 5. 行动：调用 multiply 工具
# 6. 观察：得到 1200万
# 7. 回答：公司去年营收1000万，增长20%后预计1200万
```

### 10.3 OpenAI Function Calling Agent

```python
from llama_index.agent.openai import OpenAIAgent

# OpenAI Function Calling Agent（推荐 GPT-4/3.5-turbo）
agent = OpenAIAgent.from_tools(
    tools=[query_tool, multiply_tool, add_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True,
    system_prompt="你是一个有用的助手，能够查询知识库并进行数学计算。",
)

response = agent.chat("帮我查一下产品A的价格，然后计算买5个需要多少钱")
```

### 10.4 多文档 Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 为每个文档创建独立的查询引擎和工具
tools = []
for doc_name, doc_index in document_indices.items():
    tool = QueryEngineTool.from_defaults(
        query_engine=doc_index.as_query_engine(),
        name=f"doc_{doc_name}",
        description=f"包含 {doc_name} 的详细信息",
    )
    tools.append(tool)

# 创建可以跨多个文档检索的 Agent
agent = ReActAgent.from_tools(
    tools=tools,
    verbose=True,
    max_iterations=15,
)

response = agent.chat("对比文档A和文档B中关于定价策略的不同")
```

---

## 十一、高级特性

### 11.1 Prompt 自定义

```python
from llama_index.core import PromptTemplate

# ===== 自定义 QA Prompt =====
qa_prompt_tmpl = PromptTemplate(
    """\
以下是上下文信息：
---------------------
{context_str}
---------------------

请根据以上上下文信息（而非先验知识）回答以下问题。
如果上下文中没有相关信息，请回答"根据提供的资料，我无法回答这个问题"。
请用中文回答，并尽可能详细。

问题：{query_str}

回答："""
)

# ===== 自定义 Refine Prompt =====
refine_prompt_tmpl = PromptTemplate(
    """\
原始问题如下：{query_str}
已有的回答如下：{existing_answer}
我们有机会通过以下新的上下文来优化现有回答：
---------------------
{context_msg}
---------------------
请根据新的上下文优化原始回答，使其更准确完整。
如果新上下文没有帮助，请返回原始回答。

优化后的回答："""
)

# 应用到查询引擎
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt_tmpl,
    refine_template=refine_prompt_tmpl,
)

# ===== 查看默认 Prompt =====
prompts = query_engine.get_prompts()
for key, prompt in prompts.items():
    print(f"Prompt Key: {key}")
    print(f"Template: {prompt.get_template()}")
    print("---")

# ===== 动态更新 Prompt =====
query_engine.update_prompts({
    "response_synthesizer:text_qa_template": qa_prompt_tmpl,
})
```

### 11.2 Metadata Filtering（元数据过滤）

```python
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

# 在查询时进行元数据过滤
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="source",
            value="annual_report_2024.pdf",
            operator=FilterOperator.EQ,    # 等于
        ),
        MetadataFilter(
            key="page_number",
            value=10,
            operator=FilterOperator.GTE,    # 大于等于
        ),
        MetadataFilter(
            key="category",
            value=["finance", "strategy"],
            operator=FilterOperator.IN,     # 在列表中
        ),
    ],
    condition=FilterCondition.AND,  # 所有条件取 AND
)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    filters=filters,
)

# 支持的操作符
# FilterOperator.EQ    - 等于
# FilterOperator.NE    - 不等于
# FilterOperator.GT    - 大于
# FilterOperator.GTE   - 大于等于
# FilterOperator.LT    - 小于
# FilterOperator.LTE   - 小于等于
# FilterOperator.IN    - 在列表中
# FilterOperator.NIN   - 不在列表中
# FilterOperator.TEXT_MATCH - 文本匹配
```

### 11.3 Router Query Engine（路由查询引擎）

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import QueryEngineTool

# 创建不同用途的查询引擎工具
tools = [
    QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="用于回答具体的事实性问题，如产品参数、数据指标等",
    ),
    QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="用于生成摘要性回答，如文档概述、趋势分析等",
    ),
]

# 单选路由器（选择一个最佳引擎）
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
    verbose=True,
)

# 多选路由器（可以同时使用多个引擎）
router_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=tools,
    verbose=True,
)

response = router_engine.query("请总结一下这份报告的主要内容")
# LLM 会判断这是摘要任务，路由到 summary_query_engine
```

### 11.4 评估（Evaluation）

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
)

# ===== 1. 忠实度评估（答案是否基于检索到的上下文）=====
faithfulness_evaluator = FaithfulnessEvaluator()

response = query_engine.query("什么是LlamaIndex？")
eval_result = faithfulness_evaluator.evaluate_response(response=response)
print(f"忠实度: {eval_result.passing}")      # True/False
print(f"得分: {eval_result.score}")          # 0-1
print(f"反馈: {eval_result.feedback}")       # 详细说明

# ===== 2. 相关性评估（答案是否与问题相关）=====
relevancy_evaluator = RelevancyEvaluator()
eval_result = relevancy_evaluator.evaluate_response(
    query="什么是LlamaIndex？",
    response=response,
)

# ===== 3. 正确性评估（需要参考答案）=====
correctness_evaluator = CorrectnessEvaluator()
eval_result = correctness_evaluator.evaluate(
    query="什么是LlamaIndex？",
    response="LlamaIndex是一个数据框架...",
    reference="LlamaIndex是一个用于构建LLM应用的数据框架...",
)

# ===== 4. 批量评估 =====
runner = BatchEvalRunner(
    evaluators={
        "faithfulness": faithfulness_evaluator,
        "relevancy": relevancy_evaluator,
    },
    workers=4,  # 并行评估
)

eval_results = await runner.aevaluate_queries(
    query_engine=query_engine,
    queries=["问题1", "问题2", "问题3"],
)
```

### 11.5 Observability（可观测性）

```python
# ===== 使用回调管理器 =====
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    TokenCountingHandler,
)
import tiktoken

# 调试处理器（打印所有事件）
debug_handler = LlamaDebugHandler(print_trace_on_end=True)

# Token 计数器
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode
)

# 设置回调管理器
callback_manager = CallbackManager([debug_handler, token_counter])
Settings.callback_manager = callback_manager

# 执行查询后查看统计
response = query_engine.query("测试问题")

# 查看 Token 使用量
print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
print(f"LLM Completion Tokens: {token_counter.completion_llm_token_count}")
print(f"Total LLM Tokens: {token_counter.total_llm_token_count}")

# 查看调试事件
for event in debug_handler.get_events():
    print(event)


# ===== 集成 Arize Phoenix（推荐的可观测性工具）=====
# pip install arize-phoenix llama-index-callbacks-arize-phoenix

import phoenix as px
from llama_index.core import set_global_handler

px.launch_app()  # 启动 Phoenix UI
set_global_handler("arize_phoenix")

# 之后所有的 LlamaIndex 操作都会被追踪
# 在 http://localhost:6006 查看详细 trace
```

### 11.6 Multi-Modal（多模态）

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

# 多模态 LLM
mm_llm = OpenAIMultiModal(
    model="gpt-4o",
    max_new_tokens=1000,
)

# 加载包含图片的文档
documents = SimpleDirectoryReader(
    input_dir="./data_with_images",
    required_exts=[".jpg", ".png", ".pdf"],
).load_data()

# 创建多模态索引
mm_index = MultiModalVectorStoreIndex.from_documents(documents)

# 多模态查询
query_engine = mm_index.as_query_engine(multi_modal_llm=mm_llm)
response = query_engine.query("描述一下图片中的内容")
```

---

## 十二、完整实战案例

### 12.1 案例一：PDF 知识库问答系统

```python
"""
完整的 PDF 知识库问答系统
"""
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

# =====================
# 1. 配置
# =====================
os.environ["OPENAI_API_KEY"] = "sk-xxx"

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

PERSIST_DIR = "./storage"

# =====================
# 2. 构建或加载索引
# =====================
def get_index():
    if os.path.exists(PERSIST_DIR):
        print("📦 从缓存加载索引...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("🔨 构建新索引...")
        documents = SimpleDirectoryReader(
            input_dir="./data/pdfs",
            recursive=True,
            required_exts=[".pdf"],
            filename_as_id=True,
        ).load_data()

        print(f"📄 加载了 {len(documents)} 个文档")

        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
        )

        # 持久化
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("💾 索引已保存")

        return index

# =====================
# 3. 自定义 Prompt
# =====================
QA_PROMPT = PromptTemplate(
    """\
你是一个专业的文档分析助手。请根据以下上下文信息回答问题。

上下文信息：
---------------------
{context_str}
---------------------

要求：
1. 仅基于提供的上下文回答，不要使用先验知识
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、条理清晰
4. 适当引用原文内容

问题：{query_str}

回答："""
)

# =====================
# 4. 创建查询引擎
# =====================
index = get_index()

query_engine = index.as_query_engine(
    similarity_top_k=5,
    text_qa_template=QA_PROMPT,
    response_mode="compact",
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.5),
    ],
    streaming=True,
)

# =====================
# 5. 交互式问答
# =====================
def main():
    print("\n🤖 PDF知识库问答系统已启动！输入 'quit' 退出。\n")

    while True:
        question = input("📝 你的问题: ").strip()

        if question.lower() in ('quit', 'exit', 'q'):
            print("👋 再见！")
            break

        if not question:
            continue

        print("\n🔍 正在检索并生成回答...\n")

        response = query_engine.query(question)

        # 流式输出
        print("📖 回答: ", end="")
        for text in response.response_gen:
            print(text, end="", flush=True)
        print("\n")

        # 显示来源
        print("📚 参考来源:")
        for i, node in enumerate(response.source_nodes, 1):
            score = node.score or 0
            source = node.metadata.get("file_name", "未知")
            page = node.metadata.get("page_label", "N/A")
            print(f"  [{i}] {source} (页{page}) - 相关度: {score:.4f}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
```

### 12.2 案例二：多数据源 RAG + Agent

```python
"""
多数据源 RAG 系统 + Agent
支持：知识库查询、数据库查询、API调用
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.llms.openai import OpenAI
import requests
from datetime import datetime

# =====================
# 1. 配置
# =====================
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

# =====================
# 2. 创建知识库查询工具
# =====================
docs = SimpleDirectoryReader("./data/knowledge_base").load_data()
kb_index = VectorStoreIndex.from_documents(docs)

kb_tool = QueryEngineTool(
    query_engine=kb_index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="knowledge_base",
        description=(
            "公司内部知识库。包含产品说明、技术文档、操作手册等。"
            "当用户询问产品功能、使用方法、技术细节时使用此工具。"
        ),
    ),
)

# =====================
# 3. 创建自定义函数工具
# =====================
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def calculate_price(
    base_price: float,
    quantity: int,
    discount_percent: float = 0
) -> str:
    """
    计算订单总价。

    Args:
        base_price: 单价
        quantity: 数量
        discount_percent: 折扣百分比（如 10 表示打9折）
    """
    total = base_price * quantity * (1 - discount_percent / 100)
    return f"单价: {base_price}元, 数量: {quantity}, 折扣: {discount_percent}%, 总价: {total:.2f}元"

def search_order(order_id: str) -> str:
    """
    查询订单状态。

    Args:
        order_id: 订单编号
    """
    # 模拟数据库查询
    mock_orders = {
        "ORD001": {"status": "已发货", "eta": "2024-03-15"},
        "ORD002": {"status": "处理中", "eta": "2024-03-20"},
    }
    if order_id in mock_orders:
        order = mock_orders[order_id]
        return f"订单 {order_id}: 状态={order['status']}, 预计到达={order['eta']}"
    return f"未找到订单 {order_id}"

# 转为工具
time_tool = FunctionTool.from_defaults(fn=get_current_time)
price_tool = FunctionTool.from_defaults(fn=calculate_price)
order_tool = FunctionTool.from_defaults(fn=search_order)

# =====================
# 4. 创建 Agent
# =====================
agent = ReActAgent.from_tools(
    tools=[kb_tool, time_tool, price_tool, order_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True,
    max_iterations=10,
    system_prompt="""\
你是一个智能客服助手，具备以下能力：
1. 查询公司知识库回答产品相关问题
2. 获取当前时间
3. 计算订单价格
4. 查询订单状态

请友好、专业地帮助用户。如果需要使用工具，请先思考应该使用哪个工具。
""",
)

# =====================
# 5. 对话
# =====================
# 单轮
response = agent.chat("产品A的价格是多少？如果我买10个，打8折，总价多少？")
print(response)

# 多轮
response1 = agent.chat("帮我查一下订单ORD001的状态")
response2 = agent.chat("现在几点了？")
response3 = agent.chat("我想了解一下你们的退货政策")  # 会查询知识库
```

### 12.3 案例三：Streamlit 聊天应用

```python
"""
基于 Streamlit 的 RAG 聊天应用
运行: streamlit run app.py
"""
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

# =====================
# 页面配置
# =====================
st.set_page_config(
    page_title="📚 智能文档助手",
    page_icon="🤖",
    layout="wide",
)

st.title("📚 智能文档助手")
st.caption("基于 LlamaIndex 的 RAG 聊天系统")

# =====================
# 侧边栏配置
# =====================
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("模型", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    top_k = st.slider("检索 Top-K", 1, 10, 5)

    if st.button("🗑️ 清除对话"):
        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()

# =====================
# 初始化
# =====================
@st.cache_resource(show_spinner="正在加载知识库...")
def init_index(_api_key, _model):
    import os
    os.environ["OPENAI_API_KEY"] = _api_key

    Settings.llm = OpenAI(model=_model, temperature=temperature)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    persist_dir = "./storage"
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        return index

if not api_key:
    st.warning("请在侧边栏输入 OpenAI API Key")
    st.stop()

index = init_index(api_key, model)

# =====================
# 聊天引擎
# =====================
if "chat_engine" not in st.session_state:
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        system_prompt="你是一个专业的文档分析助手，请基于知识库内容回答问题。",
        similarity_top_k=top_k,
        verbose=False,
    )

# =====================
# 消息历史
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是文档助手，请问有什么可以帮您的？"}
    ]

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================
# 用户输入
# =====================
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.chat_engine.stream_chat(prompt)

            # 流式输出
            response_text = st.write_stream(
                (token for token in response.response_gen)
            )

            # 显示来源
            if response.source_nodes:
                with st.expander("📚 参考来源"):
                    for i, node in enumerate(response.source_nodes, 1):
                        score = node.score or 0
                        source = node.metadata.get("file_name", "未知")
                        st.markdown(f"**[{i}]** {source} (相关度: {score:.3f})")
                        st.text(node.text[:300] + "...")
                        st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
```

---

## 十三、性能优化与最佳实践

### 13.1 分块策略优化

```
              分块策略选择指南
┌───────────────────────────────────────────────┐
│                                                │
│  文档类型          推荐分块策略                   │
│  ─────────        ──────────                   │
│  通用文本          SentenceSplitter             │
│                   chunk_size=512-1024           │
│                                                │
│  技术文档          SentenceSplitter             │
│                   chunk_size=1024-2048          │
│                   chunk_overlap=200-400         │
│                                                │
│  对话/日志          固定大小分割                  │
│                   chunk_size=256-512            │
│                                                │
│  语义连贯性要求高    SemanticSplitter            │
│                   基于语义相似度自动分割          │
│                                                │
│  层次化检索         HierarchicalNodeParser      │
│                   chunk_sizes=[2048, 512, 128]  │
│                                                │
└───────────────────────────────────────────────┘
```

### 13.2 检索优化策略

```python
# ===== 策略1: Hybrid Search（混合搜索）=====
# 结合向量搜索 + 关键词搜索（BM25）

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# 向量检索器
vector_retriever = index.as_retriever(similarity_top_k=5)

# BM25 检索器
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
)

# 融合检索器
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    num_queries=1,           # 不生成额外查询
    similarity_top_k=5,      # 最终返回 Top-5
    mode="reciprocal_rerank", # 使用 RRF 融合排序
)


# ===== 策略2: Sentence Window Retrieval =====
# 检索时用小窗口匹配，返回时用大窗口提供上下文

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# 使用句子窗口分割器
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,              # 前后各3个句子作为窗口
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

nodes = node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

# 查询时替换为窗口文本
query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)


# ===== 策略3: Auto-Merging Retrieval =====
# 层次化检索：如果多个子节点被检索到，自动合并为父节点

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever

# 层次分割
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

# 需要将所有节点（包含父节点）存入 DocStore
from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)
index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# 自动合并检索器
retriever = AutoMergingRetriever(
    vector_retriever=index.as_retriever(similarity_top_k=12),
    storage_context=storage_context,
    simple_ratio_thresh=0.4,  # 超过40%的子节点被检索到则合并
)


# ===== 策略4: 查询变换 =====
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE：先让 LLM 生成假设性答案，用答案去检索
hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(
    query_engine=base_query_engine,
    query_transform=hyde,
)
```

### 13.3 最佳实践清单

```
┌─────────────────────────────────────────────────────────┐
│                   最佳实践清单                            │
│                                                          │
│  📊 数据处理                                              │
│  ├── ✅ 清洗数据：去除噪音、格式化                        │
│  ├── ✅ 丰富元数据：添加 source、date、category 等        │
│  ├── ✅ 选择合适的分块大小（通常 512-1024 tokens）        │
│  ├── ✅ 保持适当的重叠（chunk_size 的 10-20%）            │
│  └── ✅ 使用 IngestionPipeline 标准化处理流程             │
│                                                          │
│  🔍 检索优化                                              │
│  ├── ✅ 使用混合搜索（向量 + BM25）                       │
│  ├── ✅ 添加重排序步骤（Cohere Rerank / Cross-Encoder）   │
│  ├── ✅ Top-K 不要太大也不要太小（通常 3-10）              │
│  ├── ✅ 设置相似度阈值过滤低质量结果                      │
│  └── ✅ 利用元数据过滤缩小搜索范围                        │
│                                                          │
│  💬 回答生成                                              │
│  ├── ✅ 自定义 Prompt 模板，给出清晰的指示               │
│  ├── ✅ 选择合适的 Response Mode                          │
│  ├── ✅ 使用流式输出提升用户体验                          │
│  └── ✅ 引导 LLM 在不确定时说"不知道"                     │
│                                                          │
│  🏗️ 工程实践                                              │
│  ├── ✅ 持久化索引，避免重复构建                          │
│  ├── ✅ 使用外部向量数据库（生产环境）                    │
│  ├── ✅ 添加评估流程，持续监控质量                        │
│  ├── ✅ 集成可观测性工具（Phoenix/Langfuse）              │
│  ├── ✅ 缓存 Embedding 结果                              │
│  └── ✅ 异步 API 调用提升吞吐量                           │
│                                                          │
│  ⚠️ 常见陷阱                                              │
│  ├── ❌ 不要用太大的 chunk_size（超出上下文窗口）          │
│  ├── ❌ 不要忽略元数据（它对检索质量影响很大）            │
│  ├── ❌ 不要盲目使用默认配置                              │
│  ├── ❌ 不要跳过评估环节                                  │
│  └── ❌ 不要在生产环境使用内存存储                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 十四、与 LangChain 的对比

| 维度           | LlamaIndex                  | LangChain            |
| -------------- | --------------------------- | -------------------- |
| **定位**       | 数据框架，专注 RAG          | 通用 LLM 应用框架    |
| **核心优势**   | 数据索引和检索              | 链式调用和组合       |
| **上手难度**   | ⭐⭐ 较简单                 | ⭐⭐⭐ 中等          |
| **RAG 能力**   | ⭐⭐⭐⭐⭐ 极强             | ⭐⭐⭐⭐ 强          |
| **Agent 能力** | ⭐⭐⭐⭐ 强                 | ⭐⭐⭐⭐⭐ 极强      |
| **数据连接器** | 160+ 种                     | 100+ 种              |
| **索引类型**   | 丰富（向量/树/图/关键词..） | 主要是向量           |
| **可定制性**   | 高                          | 极高                 |
| **生态系统**   | 成长中                      | 非常成熟             |
| **适合场景**   | 知识库问答、文档分析        | 复杂工作流、多步推理 |

> **建议**：
>
> - 如果你的核心需求是 **RAG / 文档问答** → 优先选择 **LlamaIndex**
> - 如果你需要构建 **复杂的 Agent 工作流** → 优先选择 **LangChain** 或 **LangGraph**
> - 两者可以 **互补使用**：用 LlamaIndex 做数据索引，用 LangChain 做工作流编排

```python
# LlamaIndex 和 LangChain 互操作示例
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# 将 LlamaIndex 查询引擎转为 LangChain Tool
from llama_index.core.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)

tool_config = IndexToolConfig(
    query_engine=index.as_query_engine(),
    name="KnowledgeBase",
    description="用于查询知识库",
)
langchain_tool = LlamaIndexTool.from_tool_config(tool_config)

# 然后在 LangChain Agent 中使用
```

---

## 十五、常见问题与排错

### Q1: 索引构建很慢怎么办？

```python
# 方案1: 使用批量 Embedding（减少 API 调用次数）
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=100,  # 批量大小
)

# 方案2: 使用本地 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cuda",
)

# 方案3: 使用 IngestionPipeline 缓存
pipeline = IngestionPipeline(
    transformations=[...],
    cache=IngestionCache(),
)

# 方案4: 异步并行处理
import asyncio
nodes = await pipeline.arun(documents=documents)
```

### Q2: 检索质量差怎么办？

```python
# 排查步骤：
# 1. 先检查分块是否合理
for node in index.docstore.docs.values():
    print(f"Chunk size: {len(node.text)}")
    print(f"Text preview: {node.text[:100]}")
    print("---")

# 2. 检查检索结果
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve("你的查询")
for node in nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}")
    print("---")

# 3. 优化策略
# - 调整 chunk_size（试试更小或更大的值）
# - 增加 chunk_overlap
# - 添加重排序
# - 使用混合搜索
# - 丰富元数据
# - 尝试不同的 Embedding 模型
```

### Q3: Token 消耗太高怎么办？

```python
# 方案1: 减少 Top-K
query_engine = index.as_query_engine(similarity_top_k=3)

# 方案2: 使用 COMPACT 模式（默认，减少 LLM 调用次数）
query_engine = index.as_query_engine(response_mode="compact")

# 方案3: 使用更小的 chunk_size
Settings.chunk_size = 512

# 方案4: 使用更便宜的模型做初步筛选
Settings.llm = OpenAI(model="gpt-3.5-turbo")  # 便宜
# 关键问题再用 GPT-4
```

### Q4: 如何处理中文文档？

```python
# 1. 使用中文 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文优化
)

# 2. 调整分块参数（中文通常需要更大的 chunk_size）
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

# 3. 自定义中文 Prompt
QA_PROMPT = PromptTemplate(
    "请用中文回答...\n上下文: {context_str}\n问题: {query_str}\n回答:"
)

# 4. PDF 中文提取优化
# pip install pdfplumber
from llama_index.readers.file import PDFReader
```

---

## 十六、总结与学习路线

### 学习路线图

```
                    LlamaIndex 学习路线

Level 1 - 入门 (1-2天)
├── 理解 RAG 概念
├── 安装和基础配置
├── 5行代码实现第一个 RAG
└── 理解 Document、Node、Index 概念

Level 2 - 进阶 (3-5天)
├── 掌握不同索引类型及适用场景
├── 自定义 Prompt 模板
├── 使用不同的数据加载器
├── 配置外部向量数据库
├── 使用 Chat Engine 实现多轮对话
└── Node PostProcessors 和重排序

Level 3 - 高级 (1-2周)
├── Agent 和 Tools 开发
├── 混合搜索和高级检索策略
├── Ingestion Pipeline 定制
├── 评估框架使用
├── 可观测性集成
└── 多模态支持

Level 4 - 专家 (持续)
├── 自定义 Retriever / Query Engine
├── 生产环境部署优化
├── 与 LangChain/LangGraph 集成
├── 构建复杂的多 Agent 系统
├── 性能调优和成本优化
└── 贡献开源社区
```

### 核心要点回顾

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  📌 LlamaIndex = 数据 + LLM 的桥梁                      │
│                                                          │
│  📌 核心流程: Load → Transform → Embed → Index → Query   │
│                                                          │
│  📌 最常用: VectorStoreIndex + as_query_engine()         │
│                                                          │
│  📌 质量关键: 分块策略 + Embedding 选择 + 检索优化        │
│                                                          │
│  📌 生产就绪: 外部向量库 + 持久化 + 评估 + 可观测性      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 参考资源

| 资源                   | 链接                                                                  |
| ---------------------- | --------------------------------------------------------------------- |
| 官方文档               | https://docs.llamaindex.ai                                            |
| GitHub                 | https://github.com/run-llama/llama_index                              |
| LlamaHub（数据连接器） | https://llamahub.ai                                                   |
| Discord 社区           | https://discord.gg/llamaindex                                         |
| 官方博客               | https://blog.llamaindex.ai                                            |
| 示例 Notebooks         | https://github.com/run-llama/llama_index/tree/main/docs/docs/examples |

---
