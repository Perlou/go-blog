+++
date = '2024-11-05T11:00:00+08:00'
draft = false
title = '向量数据库在AI应用中的实践'
image = '/images/covers/vector-database.png'
categories = ['AI开发']
tags = ['向量数据库', 'RAG', '数据工程']
+++

向量数据库（Vector Database）是 AI 应用的重要基础设施，特别是在 RAG（检索增强生成）场景中不可或缺。本文介绍向量数据库的原理和实践。

## 什么是向量数据库

向量数据库专门用于存储和检索高维向量数据，提供快速的相似度搜索能力。

### 核心概念

**Embedding（嵌入）**：将文本、图片等数据转换为固定维度的向量

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding = model.encode("这是一段文本")
# 输出：[0.123, -0.456, 0.789, ...] (384维)
```

**相似度计算**：

- 余弦相似度
- 欧氏距离
- 点积

## 主流向量数据库

### 1. Chroma

轻量级、易用：

````python
import chromadb
from chromadb.config import Settings

# 初始化
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# 创建集合
collection = client.create_collection(name="documents")

# 添加文档
collection.add(
    documents=["这是第一段文本", "这是第二段文本"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

# 查询
results = collection.query(
    query_texts=["相关的文本"],
    n_results=2
)


### 2. Pinecone

云原生、可扩展：

```python
import pinecone

# 初始化
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# 创建索引
pinecone.create_index("documents", dimension=384, metric="cosine")

# 连接索引
index = pinecone.Index("documents")

# 插入向量
index.upsert(vectors=[
    ("id1", embedding1, {"text": "文本1"}),
    ("id2", embedding2, {"text": "文本2"})
])

# 查询
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)


### 3. Milvus

高性能、分布式：

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接
connections.connect("default", host="localhost", port="19530")

# 定义schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields)

# 创建集合
collection = Collection("documents", schema)

# 插入数据
entities = [
    [1, 2, 3],  # ids
    [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],  # embeddings
    ["文本1", "文本2", "文本3"]  # texts
]
collection.insert(entities)

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# 查询
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=5
)


## 实战：构建文档问答系统

### 1. 文档处理

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# 加载PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_documents(pages)


### 2. 生成 Embedding

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 批量生成
texts = [chunk.page_content for chunk in chunks]
vectors = embeddings.embed_documents(texts)


### 3. 存储到向量数据库

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


### 4. 检索与问答

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 提问
result = qa_chain({"query": "这个文档的主要内容是什么？"})
print(result['result'])
print(result['source_documents'])


## 性能优化

### 1. 索引优化

```python
# Milvus IVF索引配置
index_params = {
    "index_type": "IVF_SQ8",  # 量化压缩
    "metric_type": "IP",      # 内积
    "params": {
        "nlist": 1024          # 聚类数量
    }
}


### 2. 查询优化

```python
# 使用过滤器
results = vectorstore.similarity_search(
    query="查询文本",
    k=5,
    filter={"source": "特定来源"}
)

# 混合搜索（向量 + 关键词）
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


### 3. 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    return embeddings.embed_query(text)


## 评估与监控

### 1. 检索质量评估

```python
def evaluate_retrieval(queries, ground_truth):
    """
    评估检索准确性
    queries: 测试查询列表
    ground_truth: 每个查询的正确文档ID
    """
    precision_scores = []
    recall_scores = []

    for query, truth_ids in zip(queries, ground_truth):
        results = vectorstore.similarity_search(query, k=5)
        retrieved_ids = [r.metadata['id'] for r in results]

        # 计算precision和recall
        hits = len(set(retrieved_ids) & set(truth_ids))
        precision = hits / len(retrieved_ids)
        recall = hits / len(truth_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        'avg_precision': sum(precision_scores) / len(precision_scores),
        'avg_recall': sum(recall_scores) / len(recall_scores)
    }


### 2. 性能监控

```python
import time

class VectorStoreMonitor:
    def __init__(self):
        self.query_times = []
        self.query_count = 0

    def track_query(self, query_func, *args, **kwargs):
        start = time.time()
        result = query_func(*args, **kwargs)
        duration = time.time() - start

        self.query_times.append(duration)
        self.query_count += 1

        return result

    def get_stats(self):
        return {
            'total_queries': self.query_count,
            'avg_latency': sum(self.query_times) / len(self.query_times),
            'p95_latency': sorted(self.query_times)[int(len(self.query_times) * 0.95)]
        }


## 最佳实践

1. **选择合适的 chunk size**：根据内容特点调整
2. **使用元数据过滤**：提高检索精度
3. **定期更新索引**：保持数据新鲜度
4. **监控性能**：追踪查询延迟和准确性
5. **备份策略**：定期备份向量数据
6. **多语言支持**：选择支持多语言的 embedding 模型

## 常见问题

**Q: 如何处理大规模数据？**
A: 使用分布式向量数据库（如 Milvus），启用分片和副本

**Q: 如何提高检索准确性？**
A: 优化 chunk 策略、使用更好的 embedding 模型、结合关键词检索

**Q: 向量维度如何选择？**
A: 权衡性能和效果，通常 384-1536 维，更高维度效果更好但成本更高

## 总结

向量数据库是构建 AI 应用的基础设施。选择合适的向量数据库，优化 embedding 和检索策略，可以构建高效的语义搜索和 RAG 系统。关键是根据实际需求平衡性能、成本和效果。
````
